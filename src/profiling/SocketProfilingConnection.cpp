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
    // Dummy return value, function not implemented
    return true;
}

void SocketProfilingConnection::Close()
{
    // Function not implemented
}

bool SocketProfilingConnection::WritePacket(const char* buffer, uint32_t length)
{
    // Dummy return value, function not implemented
    return true;
}

Packet SocketProfilingConnection::ReadPacket(uint32_t timeout)
{
    // Dummy return value, function not implemented
    return {472580096, 0, nullptr};
}

} // namespace profiling
} // namespace armnn
