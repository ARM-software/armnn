//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SocketProfilingConnection.hpp"

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <cerrno>
#include <string>

namespace armnn
{
namespace profiling
{

SocketProfilingConnection::SocketProfilingConnection()
{
    memset(m_Socket, 0, sizeof(m_Socket));
    // Note: we're using Linux specific SOCK_CLOEXEC flag.
    m_Socket[0].fd = socket(PF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (m_Socket[0].fd == -1)
    {
        throw RuntimeException(std::string("Socket construction failed: ") + strerror(errno));
    }

    // Connect to the named unix domain socket.
    struct sockaddr_un server{};
    memset(&server, 0, sizeof(sockaddr_un));
    // As m_GatorNamespace begins with a null character we need to ignore that when getting its length.
    memcpy(server.sun_path, m_GatorNamespace, strlen(m_GatorNamespace + 1) + 1);
    server.sun_family = AF_UNIX;
    if (0 != connect(m_Socket[0].fd, reinterpret_cast<const sockaddr*>(&server), sizeof(sockaddr_un)))
    {
        close(m_Socket[0].fd);
        throw RuntimeException(std::string("Cannot connect to stream socket: ") + strerror(errno));
    }

    // Our socket will only be interested in polling reads.
    m_Socket[0].events = POLLIN;

    // Make the socket non blocking.
    const int currentFlags = fcntl(m_Socket[0].fd, F_GETFL);
    if (0 != fcntl(m_Socket[0].fd, F_SETFL, currentFlags | O_NONBLOCK))
    {
        close(m_Socket[0].fd);
        throw RuntimeException(std::string("Failed to set socket as non blocking: ") + strerror(errno));
    }
}

bool SocketProfilingConnection::IsOpen() const
{
    return m_Socket[0].fd > 0;
}

void SocketProfilingConnection::Close()
{
    if (close(m_Socket[0].fd) != 0)
    {
        throw RuntimeException(std::string("Cannot close stream socket: ") + strerror(errno));
    }

    memset(m_Socket, 0, sizeof(m_Socket));
}

bool SocketProfilingConnection::WritePacket(const unsigned char* buffer, uint32_t length)
{
    if (buffer == nullptr || length == 0)
    {
        return false;
    }

    return write(m_Socket[0].fd, buffer, length) != -1;
}

Packet SocketProfilingConnection::ReadPacket(uint32_t timeout)
{
    // Poll for data on the socket or until timeout occurs
    int pollResult = poll(m_Socket, 1, static_cast<int>(timeout));

    switch (pollResult)
    {
    case -1: // Error
        throw RuntimeException(std::string("Read failure from socket: ") + strerror(errno));

    case 0: // Timeout
        throw RuntimeException("Timeout while reading from socket");

    default: // Normal poll return but it could still contain an error signal

        // Check if the socket reported an error
        if (m_Socket[0].revents & (POLLNVAL | POLLERR | POLLHUP))
        {
            throw Exception(std::string("Socket 0 reported an error: ") + strerror(errno));
        }

        // Check if there is data to read
        if (!(m_Socket[0].revents & (POLLIN)))
        {
            // No data to read from the socket. Silently ignore and continue
            return Packet();
        }

        // There is data to read, read the header first
        char header[8] = {};
        if (8 != recv(m_Socket[0].fd, &header, sizeof(header), 0))
        {
            // What do we do here if there's not a valid 8 byte header to read?
            throw RuntimeException("The received packet did not contains a valid MIPE header");
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
        }

        ssize_t receivedLength = recv(m_Socket[0].fd, packetData.get(), dataLength, 0);
        if (receivedLength < 0)
        {
            throw armnn::RuntimeException(std::string("Error occured on recv: ") + strerror(errno));
        }
        if (dataLength != static_cast<uint32_t>(receivedLength))
        {
            // What do we do here if we can't read in a full packet?
            throw RuntimeException("Invalid MIPE packet");
        }

        return Packet(metadataIdentifier, dataLength, packetData);
    }
}

} // namespace profiling
} // namespace armnn
