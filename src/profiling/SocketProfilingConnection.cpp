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
        throw armnn::Exception(std::string(": Socket construction failed: ") + strerror(errno));
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
        throw armnn::Exception(std::string(": Cannot connect to stream socket: ") + strerror(errno));
    }

    // Our socket will only be interested in polling reads.
    m_Socket[0].events = POLLIN;

    // Make the socket non blocking.
    const int currentFlags = fcntl(m_Socket[0].fd, F_GETFL);
    if (0 != fcntl(m_Socket[0].fd, F_SETFL, currentFlags | O_NONBLOCK))
    {
        close(m_Socket[0].fd);
        throw armnn::Exception(std::string(": Failed to set socket as non blocking: ") + strerror(errno));
    }
}

bool SocketProfilingConnection::IsOpen()
{
    if (m_Socket[0].fd > 0)
    {
        return true;
    }
    return false;
}

void SocketProfilingConnection::Close()
{
    if (0 == close(m_Socket[0].fd))
    {
        memset(m_Socket, 0, sizeof(m_Socket));
    }
    else
    {
        throw armnn::Exception(std::string(": Cannot close stream socket: ") + strerror(errno));
    }
}

bool SocketProfilingConnection::WritePacket(const char* buffer, uint32_t length)
{
    if (-1 == write(m_Socket[0].fd, buffer, length))
    {
        return false;
    }
    return true;
}

Packet SocketProfilingConnection::ReadPacket(uint32_t timeout)
{
    // Poll for data on the socket or until timeout.
    int pollResult = poll(m_Socket, 1, static_cast<int>(timeout));
    if (pollResult > 0)
    {
        // Normal poll return but it could still contain an error signal.
        if (m_Socket[0].revents & (POLLNVAL | POLLERR | POLLHUP))
        {
            throw armnn::Exception(std::string(": Read failure from socket: ") + strerror(errno));
        }
        else if (m_Socket[0].revents & (POLLIN)) // There is data to read.
        {
            // Read the header first.
            char header[8];
            if (8 != recv(m_Socket[0].fd, &header, sizeof header, 0))
            {
                // What do we do here if there's not a valid 8 byte header to read?
                throw armnn::Exception(": Received packet did not contains a valid MIPE header. ");
            }
            // stream_metadata_identifier is the first 4 bytes.
            uint32_t metadataIdentifier = static_cast<uint32_t>(header[0]) << 24 |
                                          static_cast<uint32_t>(header[1]) << 16 |
                                          static_cast<uint32_t>(header[2]) << 8  |
                                          static_cast<uint32_t>(header[3]);
            // data_length is the next 4 bytes.
            uint32_t dataLength = static_cast<uint32_t>(header[4]) << 24 |
                                  static_cast<uint32_t>(header[5]) << 16 |
                                  static_cast<uint32_t>(header[6]) << 8  |
                                  static_cast<uint32_t>(header[7]);

            std::unique_ptr<char[]> packetData;
            if (dataLength > 0)
            {
                packetData = std::make_unique<char[]>(dataLength);
            }

            if (dataLength != recv(m_Socket[0].fd, packetData.get(), dataLength, 0))
            {
                // What do we do here if we can't read in a full packet?
                throw armnn::Exception(": Invalid MIPE packet.");
            }
            return {metadataIdentifier, dataLength, packetData};
        }
        else // Some unknown return signal.
        {
            throw armnn::Exception(": Poll returned an unexpected event." );
        }
    }
    else if (pollResult == -1)
    {
        throw armnn::Exception(std::string(": Read failure from socket: ") + strerror(errno));
    }
    else // it's 0 so a timeout.
    {
        throw armnn::TimeoutException(": Timeout while reading from socket.");
    }
}

} // namespace profiling
} // namespace armnn
