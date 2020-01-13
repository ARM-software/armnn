//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SocketProfilingConnection.hpp"

#include <cerrno>
#include <fcntl.h>
#include <string>

using namespace armnnUtils;

namespace armnn
{
namespace profiling
{

SocketProfilingConnection::SocketProfilingConnection()
{
    Sockets::Initialize();
    memset(m_Socket, 0, sizeof(m_Socket));
    // Note: we're using Linux specific SOCK_CLOEXEC flag.
    m_Socket[0].fd = socket(PF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (m_Socket[0].fd == -1)
    {
        throw armnn::RuntimeException(std::string("Socket construction failed: ") + strerror(errno));
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
        throw armnn::RuntimeException(std::string("Cannot connect to stream socket: ") + strerror(errno));
    }

    // Our socket will only be interested in polling reads.
    m_Socket[0].events = POLLIN;

    // Make the socket non blocking.
    if (!Sockets::SetNonBlocking(m_Socket[0].fd))
    {
        Close();
        throw armnn::RuntimeException(std::string("Failed to set socket as non blocking: ") + strerror(errno));
    }
}

bool SocketProfilingConnection::IsOpen() const
{
    return m_Socket[0].fd > 0;
}

void SocketProfilingConnection::Close()
{
    if (Sockets::Close(m_Socket[0].fd) != 0)
    {
        throw armnn::RuntimeException(std::string("Cannot close stream socket: ") + strerror(errno));
    }

    memset(m_Socket, 0, sizeof(m_Socket));
}

bool SocketProfilingConnection::WritePacket(const unsigned char* buffer, uint32_t length)
{
    if (buffer == nullptr || length == 0)
    {
        return false;
    }

    return Sockets::Write(m_Socket[0].fd, buffer, length) != -1;
}

Packet SocketProfilingConnection::ReadPacket(uint32_t timeout)
{
    // Is there currently at least a header worth of data waiting to be read?
    int bytes_available = 0;
    Sockets::Ioctl(m_Socket[0].fd, FIONREAD, &bytes_available);
    if (bytes_available >= 8)
    {
        // Yes there is. Read it:
        return ReceivePacket();
    }

    // Poll for data on the socket or until timeout occurs
    int pollResult = Sockets::Poll(&m_Socket[0], 1, static_cast<int>(timeout));

    switch (pollResult)
    {
    case -1: // Error
        throw armnn::RuntimeException(std::string("Read failure from socket: ") + strerror(errno));

    case 0: // Timeout
        throw TimeoutException("Timeout while reading from socket");

    default: // Normal poll return but it could still contain an error signal
        // Check if the socket reported an error
        if (m_Socket[0].revents & (POLLNVAL | POLLERR | POLLHUP))
        {
            if (m_Socket[0].revents == POLLNVAL)
            {
                // This is an unrecoverable error.
                Close();
                throw armnn::RuntimeException(std::string("Error while polling receiving socket: POLLNVAL"));
            }
            if (m_Socket[0].revents == POLLERR)
            {
                throw armnn::RuntimeException(std::string("Error while polling receiving socket: POLLERR: ") +
                                              strerror(errno));
            }
            if (m_Socket[0].revents == POLLHUP)
            {
                // This is an unrecoverable error.
                Close();
                throw armnn::RuntimeException(std::string("Connection closed by remote client: POLLHUP"));
            }
        }

        // Check if there is data to read
        if (!(m_Socket[0].revents & (POLLIN)))
        {
            // This is a corner case. The socket as been woken up but not with any data.
            // We'll throw a timeout exception to loop around again.
            throw armnn::TimeoutException("File descriptor was polled but no data was available to receive.");
        }

        return ReceivePacket();
    }
}

Packet SocketProfilingConnection::ReceivePacket()
{
    char header[8] = {};
    long receiveResult = Sockets::Read(m_Socket[0].fd, &header, sizeof(header));
    // We expect 8 as the result here. 0 means EOF, socket is closed. -1 means there been some other kind of error.
    switch( receiveResult )
    {
        case 0:
            // Socket has closed.
            Close();
            throw armnn::RuntimeException("Remote socket has closed the connection.");
        case -1:
            // There's been a socket error. We will presume it's unrecoverable.
            Close();
            throw armnn::RuntimeException(std::string("Error occured on recv: ") + strerror(errno));
        default:
            if (receiveResult < 8)
            {
                throw armnn::RuntimeException("The received packet did not contains a valid MIPE header");
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
        long receivedLength = Sockets::Read(m_Socket[0].fd, packetData.get(), dataLength);
        if (receivedLength < 0)
        {
            throw armnn::RuntimeException(std::string("Error occurred on recv: ") + strerror(errno));
        }
        if (dataLength != static_cast<uint32_t>(receivedLength))
        {
            // What do we do here if we can't read in a full packet?
            throw armnn::RuntimeException("Invalid MIPE packet");
        }
    }

    return Packet(metadataIdentifier, dataLength, packetData);
}

} // namespace profiling
} // namespace armnn
