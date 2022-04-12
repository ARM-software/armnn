//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

// This file (along with its corresponding .cpp) defines a very thin platform abstraction layer for the use of
// networking sockets. Thankfully the underlying APIs on Windows and Linux are very similar so not much conversion
// is needed (typically just forwarding the parameters to a differently named function).
// Some of the APIs are in fact completely identical and so no forwarding function is needed.

#if !defined(ARMNN_DISABLE_SOCKETS)

#pragma once

#if defined(__unix__) || defined(__APPLE__)
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/un.h>
#elif defined(_MSC_VER)
#include <WindowsWrapper.hpp>
#include <winsock2.h>
#include <afunix.h>
#elif defined(__MINGW32__)
#include <WindowsWrapper.hpp>
#include <winsock2.h>
#endif

namespace arm
{
namespace pipe
{

#if defined(__unix__)

using Socket = int;
using PollFd = pollfd;

#elif defined(__APPLE__)

using Socket = int;
using PollFd = pollfd;
#define SOCK_CLOEXEC 0

#elif defined(_MSC_VER)

using Socket = SOCKET;
using PollFd = WSAPOLLFD;
using nfds_t = int;
using socklen_t = int;
#define SOCK_CLOEXEC 0

#elif defined(__MINGW32__)

using Socket = SOCKET;
using PollFd = WSAPOLLFD;
using nfds_t = int;
using socklen_t = int;
#define SOCK_CLOEXEC 0

#endif

/// Performs any required one-time setup.
bool Initialize();

int Close(Socket s);

bool SetNonBlocking(Socket s);

long Write(Socket s, const void* buf, size_t len);

long Read(Socket s, void* buf, size_t len);

int Ioctl(Socket s, unsigned long int cmd, void* arg);

int Poll(PollFd* fds, nfds_t numFds, int timeout);

Socket Accept(Socket s, sockaddr* addr, socklen_t* addrlen, int flags);

} // namespace arm
} // namespace pipe

#endif
