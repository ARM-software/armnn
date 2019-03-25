//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "armnn/TypesUtils.hpp"

#include <vector>

namespace armnn
{

enum class LogSeverity
{
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Fatal
};

/// Configures the logging behaviour of the ARMNN library.
///     printToStandardOutput: Set to true if log messages should be printed to the standard output.
///     printToDebugOutput: Set to true if log messages be printed to a platform-specific debug output
///       (where supported).
///     severity: All log messages that are at this severity level or higher will be printed, others will be ignored.
void ConfigureLogging(bool printToStandardOutput, bool printToDebugOutput, LogSeverity severity);

} // namespace armnn
