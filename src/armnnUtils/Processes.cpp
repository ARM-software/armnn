//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Processes.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#elif defined(_MSC_VER)
#include <common/include/WindowsWrapper.hpp>
#endif

namespace armnnUtils
{
namespace Processes
{

int GetCurrentId()
{
#if defined(__unix__) || defined(__APPLE__)
    return getpid();
#elif defined(_MSC_VER)
    return ::GetCurrentProcessId();
#endif
}

}
}
