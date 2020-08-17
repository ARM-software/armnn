//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <common/include/NetworkSockets.hpp>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#include <fcntl.h>
#endif

#include <common/include/Conversion.hpp>
#include <common/include/IgnoreUnused.hpp>
#include <common/include/NumericCast.hpp>

namespace arm
{
namespace pipe
{

bool Initialize()
{
#if defined(__unix__) || defined(__APPLE__)
    return true;
#elif defined(_MSC_VER) || defined(__MINGW32__)
    WSADATA wsaData;
    return WSAStartup(MAKEWORD(2, 2), &wsaData) == 0;
#endif
}

int Close(Socket s)
{
#if defined(__unix__) || defined(__APPLE__)
    return close(s);
#elif defined(_MSC_VER) || defined(__MINGW32__)
    return closesocket(s);
#endif
}


bool SetNonBlocking(Socket s)
{
#if defined(__unix__) || defined(__APPLE__)
    const int currentFlags = fcntl(s, F_GETFL);
    return fcntl(s, F_SETFL, currentFlags | O_NONBLOCK) == 0;
#elif defined(_MSC_VER)
    u_long mode = 1;
    return ioctlsocket(s, FIONBIO, &mode) == 0;
#elif defined(__MINGW32__)
    u_long mode = 1;
    return ioctlsocket(s, arm::pipe::numeric_cast<long>(FIONBIO), &mode) == 0;
#endif
}


long Write(Socket s, const void* buf, size_t len)
{
#if defined(__unix__) || defined(__APPLE__)
    return write(s, buf, len);
#elif defined(_MSC_VER) || defined(__MINGW32__)
    return send(s, static_cast<const char*>(buf), static_cast<int>(len), 0);
#endif
}


long Read(Socket s, void* buf, size_t len)
{
#if defined(__unix__) || defined(__APPLE__)
    return read(s, buf, len);
#elif defined(_MSC_VER) || defined(__MINGW32__)
    return recv(s, static_cast<char*>(buf), static_cast<int>(len), 0);
#endif
}

int Ioctl(Socket s, unsigned long int cmd, void* arg)
{
#if defined(__unix__) || defined(__APPLE__)
    ARM_PIPE_NO_CONVERSION_WARN_BEGIN
    return ioctl(s, static_cast<int>(cmd), arg);
    ARM_PIPE_NO_CONVERSION_WARN_END
#elif defined(_MSC_VER) || defined(__MINGW32__)
    ARM_PIPE_NO_CONVERSION_WARN_BEGIN
    return ioctlsocket(s, cmd, static_cast<u_long*>(arg));
    ARM_PIPE_NO_CONVERSION_WARN_END
#endif
}


int Poll(PollFd* fds, nfds_t numFds, int timeout)
{
#if defined(__unix__) || defined(__APPLE__)
    return poll(fds, numFds, timeout);
#elif defined(_MSC_VER) || defined(__MINGW32__)
    return WSAPoll(fds, arm::pipe::numeric_cast<unsigned long>(numFds), timeout);
#endif
}


arm::pipe::Socket Accept(Socket s, sockaddr* addr, socklen_t* addrlen, int flags)
{
#if defined(__unix__)
    return accept4(s, addr, addrlen, flags);
#elif defined(__APPLE__)
    IgnoreUnused(flags);
    return accept(s, addr, addrlen);
#elif defined(_MSC_VER) || defined(__MINGW32__)
    IgnoreUnused(flags);
    return accept(s, addr, reinterpret_cast<int*>(addrlen));
#endif
}

} // pipe
} // arm
