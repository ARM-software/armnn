//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Processes.hpp"

#if defined(__unix__)
#include <unistd.h>
#elif defined(_MSC_VER)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

namespace armnnUtils
{
namespace Processes
{

int GetCurrentId()
{
#if defined(__unix__)
    return getpid();
#elif defined(_MSC_VER)
    return ::GetCurrentProcessId();
#endif
}

}
}
