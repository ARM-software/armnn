//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NetworkSockets.hpp"

#if defined(__unix__)
#include <unistd.h>
#include <fcntl.h>
#endif

namespace armnnUtils
{
namespace Sockets
{

bool Initialize()
{
#if defined(__unix__)
    return true;
#elif defined(_MSC_VER)
    WSADATA wsaData;
    return WSAStartup(MAKEWORD(2, 2), &wsaData) == 0;
#endif
}

int Close(Socket s)
{
#if defined(__unix__)
    return close(s);
#elif defined(_MSC_VER)
    return closesocket(s);
#endif
}


bool SetNonBlocking(Socket s)
{
#if defined(__unix__)
    const int currentFlags = fcntl(s, F_GETFL);
    return fcntl(s, F_SETFL, currentFlags | O_NONBLOCK) == 0;
#elif defined(_MSC_VER)
    u_long mode = 1;
    return ioctlsocket(s, FIONBIO, &mode) == 0;
#endif
}


long Write(Socket s, const void* buf, size_t len)
{
#if defined(__unix__)
    return write(s, buf, len);
#elif defined(_MSC_VER)
    return send(s, static_cast<const char*>(buf), len, 0);
#endif
}


long Read(Socket s, void* buf, size_t len)
{
#if defined(__unix__)
    return read(s, buf, len);
#elif defined(_MSC_VER)
    return recv(s, static_cast<char*>(buf), len, 0);
#endif
}

int Ioctl(Socket s, unsigned long int cmd, void* arg)
{
#if defined(__ANDROID__)
    return ioctl(s, static_cast<int>(cmd), arg);
#elif defined(__unix__)
    return ioctl(s, cmd, arg);
#elif defined(_MSC_VER)
    return ioctlsocket(s, cmd, static_cast<u_long*>(arg));
#endif
}


int Poll(PollFd* fds, nfds_t numFds, int timeout)
{
#if defined(__unix__)
    return poll(fds, numFds, timeout);
#elif defined(_MSC_VER)
    return WSAPoll(fds, numFds, timeout);
#endif
}


armnnUtils::Sockets::Socket Accept(Socket s, sockaddr* addr, socklen_t* addrlen, int flags)
{
#if defined(__unix__)
    return accept4(s, addr, addrlen, flags);
#elif defined(_MSC_VER)
    return accept(s, addr, reinterpret_cast<int*>(addrlen));
#endif
}

}
}
