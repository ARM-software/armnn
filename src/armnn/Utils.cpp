//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "armnn/Utils.hpp"
#include "Logging.hpp"

#include <boost/log/core.hpp>

namespace armnn
{
void ConfigureLogging(bool printToStandardOutput, bool printToDebugOutput, LogSeverity severity)
{
    using armnnUtils::ConfigureLogging;
    ConfigureLogging(boost::log::core::get().get(), printToStandardOutput, printToDebugOutput, severity);
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