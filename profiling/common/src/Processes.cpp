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

namespace arm
{
namespace pipe
{

int GetCurrentProcessId()
{
#if !defined(ARMNN_DISABLE_PROCESSES)
#if defined(__unix__) || defined(__APPLE__)
    return getpid();
#elif defined(_MSC_VER)
    return ::GetCurrentProcessId();
#endif
#else
    return 0;
#endif
}

} // namespace pipe
} // namespace arm
