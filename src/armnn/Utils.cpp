//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "armnn/Logging.hpp"
#include "armnn/Utils.hpp"
#include "armnn/Version.hpp"

#if !defined(BARE_METAL) && (defined(__arm__) || defined(__aarch64__))

#include <sys/auxv.h>
#include <asm/hwcap.h>

#endif

namespace armnn
{
void ConfigureLogging(bool printToStandardOutput, bool printToDebugOutput, LogSeverity severity)
{
    SetAllLoggingSinks(printToStandardOutput, printToDebugOutput, false);
    SetLogFilter(severity);
}

// Defaults to logging completely disabled.
// The user of the library must enable it if they want by calling armnn::ConfigureLogging().
struct DefaultLoggingConfiguration
{
    DefaultLoggingConfiguration()
    {
        ConfigureLogging(false, false, LogSeverity::Trace);
    }
};

static DefaultLoggingConfiguration g_DefaultLoggingConfiguration;

// Detect the presence of Neon on Linux
bool NeonDetected()
{
#if !defined(BARE_METAL) && (defined(__arm__) || defined(__aarch64__))
    auto hwcaps= getauxval(AT_HWCAP);
#endif

#if !defined(BARE_METAL) && defined(__aarch64__)

    if (hwcaps & HWCAP_ASIMD)
    {
        // On an arm64 device with Neon.
        return true;
    }
    else
    {
        // On an arm64 device without Neon.
        return false;
    }

#endif
#if !defined(BARE_METAL) && defined(__arm__)

    if (hwcaps & HWCAP_NEON)
    {
        // On an armhf device with Neon.
        return true;
    }
    else
    {
        // On an armhf device without Neon.
        return false;
    }

#endif

    // This method of Neon detection is only supported on Linux so in order to prevent a false negative
    // we will return true in cases where detection did not run.
    return true;
}

const std::string GetVersion()
{
    return ARMNN_VERSION;
}

} // namespace armnn