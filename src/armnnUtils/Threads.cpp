//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnnUtils/Threads.hpp>

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)
#elif defined(_MSC_VER)
#include <common/include/WindowsWrapper.hpp>
#elif defined(__APPLE__)
#include "AvailabilityMacros.h"
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
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
#elif defined(__APPLE__)
    uint64_t threadId;
    int iRet = pthread_threadid_np(NULL, &threadId);
    if (iRet != 0)
    {
        return 0;
    }
    return static_cast<int>(threadId);
#endif
}

}
}
