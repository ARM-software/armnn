//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SocketProfilingConnection.hpp"

#include "common/include/SocketConnectionException.hpp"

#include <cerrno>
#include <fcntl.h>
#include <string>


namespace armnn
{
namespace profiling
{

SocketProfilingConnection::SocketProfilingConnection()
{
    arm::pipe::Initialize();
    memset(m_Socket, 0, sizeof(m_Socket));
    // Note: we're using Linux specific SOCK_CLOEXEC flag.
    m_Socket[0].fd = socket(PF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (m_Socket[0].fd == -1)
    {
        throw arm::pipe::SocketConnectionException(
            std::string("SocketProfilingConnection: Socket construction failed: ")  + strerror(errno),
            m_Socket[0].fd,
            errno);
    }

    // Connect to the named unix domain socket.
    sockaddr_un server{};
    memset(&server, 0, sizeof(sockaddr_un));
    // As m_GatorNamespace begins with a null character we need to ignore that when getting its length.
    memcpy(server.sun_path, m_GatorNamespace, strlen(m_GatorNamespace + 1) + 1);
    server.sun_family = AF_UNIX;
    if (0 != connect(m_Socket[0].fd, reinterpret_cast<const sockaddr*>(&server), sizeof(sockaddr_un)))
    {
        Close();
        throw arm::pipe::SocketConnectionException(
            std::string("SocketProfilingConnection: Cannot connect to stream socket: ")  + strerror(errno),
            m_Socket[0].fd,
            errno);
    }

    // Our socket will only be interested in polling reads.
    m_Socket[0].events = POLLIN;

    // Make the socket non blocking.
    if (!arm::pipe::SetNonBlocking(m_Socket[0].fd))
    {
        Close();
        throw arm::pipe::SocketConnectionException(
            std::string("SocketProfilingConnection: Failed to set socket as non blocking: ")  + strerror(errno),
            m_Socket[0].fd,
            errno);
    }
}

bool SocketProfilingConnection::IsOpen() const
{
    return m_Socket[0].fd > 0;
}

void SocketProfilingConnection::Close()
{
    if (arm::pipe::Close(m_Socket[0].fd) != 0)
    {
        throw arm::pipe::SocketConnectionException(
            std::string("SocketProfilingConnection: Cannot close stream socket: ")  + strerror(errno),
            m_Socket[0].fd,
            errno);
    }

    memset(m_Socket, 0, sizeof(m_Socket));
}

bool SocketProfilingConnection::WritePacket(const unsigned char* buffer, uint32_t length)
{
    if (buffer == nullptr || length == 0)
    {
        return false;
    }

    return arm::pipe::Write(m_Socket[0].fd, buffer, length) != -1;
}

arm::pipe::Packet SocketProfilingConnection::ReadPacket(uint32_t timeout)
{
    // Is there currently at least a header worth of data waiting to be read?
    int bytes_available = 0;
    arm::pipe::Ioctl(m_Socket[0].fd, FIONREAD, &bytes_available);
    if (bytes_available >= 8)
    {
        // Yes there is. Read it:
        return ReceivePacket();
    }

    // Poll for data on the socket or until timeout occurs
    int pollResult = arm::pipe::Poll(&m_Socket[0], 1, static_cast<int>(timeout));

    switch (pollResult)
    {
    case -1: // Error
        throw arm::pipe::SocketConnectionException(
            std::string("SocketProfilingConnection: Error occured while reading from socket: ") + strerror(errno),
            m_Socket[0].fd,
            errno);

    case 0: // Timeout
        throw arm::pipe::TimeoutException("SocketProfilingConnection: Timeout while reading from socket");

    default: // Normal poll return but it could still contain an error signal
        // Check if the socket reported an error
        if (m_Socket[0].revents & (POLLNVAL | POLLERR | POLLHUP))
        {
            if (m_Socket[0].revents == POLLNVAL)
            {
                // This is an unrecoverable error.
                Close();
                throw arm::pipe::SocketConnectionException(
                    std::string("SocketProfilingConnection: Error occured while polling receiving socket: POLLNVAL."),
                    m_Socket[0].fd);
            }
            if (m_Socket[0].revents == POLLERR)
            {
                throw arm::pipe::SocketConnectionException(
                    std::string(
                        "SocketProfilingConnection: Error occured while polling receiving socket: POLLERR: ")
                        + strerror(errno),
                    m_Socket[0].fd,
                    errno);
            }
            if (m_Socket[0].revents == POLLHUP)
            {
                // This is an unrecoverable error.
                Close();
                throw arm::pipe::SocketConnectionException(
                    std::string("SocketProfilingConnection: Connection closed by remote client: POLLHUP."),
                    m_Socket[0].fd);
            }
        }

        // Check if there is data to read
        if (!(m_Socket[0].revents & (POLLIN)))
        {
            // This is a corner case. The socket as been woken up but not with any data.
            // We'll throw a timeout exception to loop around again.
            throw armnn::TimeoutException(
                "SocketProfilingConnection: File descriptor was polled but no data was available to receive.");
        }

        return ReceivePacket();
    }
}

arm::pipe::Packet SocketProfilingConnection::ReceivePacket()
{
    char header[8] = {};
    long receiveResult = arm::pipe::Read(m_Socket[0].fd, &header, sizeof(header));
    // We expect 8 as the result here. 0 means EOF, socket is closed. -1 means there been some other kind of error.
    switch( receiveResult )
    {
        case 0:
            // Socket has closed.
            Close();
            throw arm::pipe::SocketConnectionException(
                std::string("SocketProfilingConnection: Remote socket has closed the connection."),
                m_Socket[0].fd);
        case -1:
            // There's been a socket error. We will presume it's unrecoverable.
            Close();
            throw arm::pipe::SocketConnectionException(
                std::string("SocketProfilingConnection: Error occured while reading the packet: ") + strerror(errno),
                m_Socket[0].fd,
                errno);
        default:
            if (receiveResult < 8)
            {
                 throw arm::pipe::SocketConnectionException(
                     std::string(
                         "SocketProfilingConnection: The received packet did not contains a valid PIPE header."),
                     m_Socket[0].fd);
            }
            break;
    }

    // stream_metadata_identifier is the first 4 bytes
    uint32_t metadataIdentifier = 0;
    std::memcpy(&metadataIdentifier, header, sizeof(metadataIdentifier));

    // data_length is the next 4 bytes
    uint32_t dataLength = 0;
    std::memcpy(&dataLength, header + 4u, sizeof(dataLength));

    std::unique_ptr<unsigned char[]> packetData;
    if (dataLength > 0)
    {
        packetData = std::make_unique<unsigned char[]>(dataLength);
        long receivedLength = arm::pipe::Read(m_Socket[0].fd, packetData.get(), dataLength);
        if (receivedLength < 0)
        {
            throw arm::pipe::SocketConnectionException(
                std::string("SocketProfilingConnection: Error occured while reading the packet: ")  + strerror(errno),
                m_Socket[0].fd,
                errno);
        }
        if (dataLength != static_cast<uint32_t>(receivedLength))
        {
            // What do we do here if we can't read in a full packet?
            throw arm::pipe::SocketConnectionException(
                std::string("SocketProfilingConnection: Invalid PIPE packet."),
                m_Socket[0].fd);
        }
    }

    return arm::pipe::Packet(metadataIdentifier, dataLength, packetData);
}

} // namespace profiling
} // namespace armnn
