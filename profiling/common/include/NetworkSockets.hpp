//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

// This file (along with its corresponding .cpp) defines a very thin platform abstraction layer for the use of
// networking sockets. Thankfully the underlying APIs on Windows and Linux are very similar so not much conversion
// is needed (typically just forwarding the parameters to a differently named function).
// Some of the APIs are in fact completely identical and so no forwarding function is needed.

#pragma once

#if defined(__unix__)
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/un.h>
#elif defined(_MSC_VER)
#include <winsock2.h>
#include <afunix.h>
#endif

namespace armnnUtils
{
namespace Sockets
{

#if defined(__unix__)

using Socket = int;
using PollFd = pollfd;

#elif defined(_MSC_VER)

using Socket = SOCKET;
using PollFd = WSAPOLLFD;
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

}
}
