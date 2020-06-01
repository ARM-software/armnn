//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "armnn/Logging.hpp"
#include "armnn/Utils.hpp"

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

} // namespace armnn