//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Threads.hpp"

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)
#elif defined(_MSC_VER)
#include "WindowsWrapper.hpp"
#endif

namespace armnnUtils
{
namespace Threads
{

int GetCurrentThreadId()
{
#if defined(__linux__)
    return static_cast<int>(gettid());
#elif defined(_MSC_VER)
    return ::GetCurrentThreadId();
#endif
}

}
}
