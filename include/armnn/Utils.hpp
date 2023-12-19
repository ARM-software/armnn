//
// Copyright © 2017,2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <string>
#include <cmath>

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


#if defined(__clang__) &&((__clang_major__>=3)||(__clang_major__==3 && __clang_minor__ >= 5))
#   define ARMNN_FALLTHROUGH [[clang::fallthrough]]
#elif defined(__GNUC__) && (__GNUC__ >= 7)
#   define ARMNN_FALLTHROUGH __attribute__((fallthrough))
#else
#   define ARMNN_FALLTHROUGH ((void)0)
#endif

bool NeonDetected();

const std::string GetVersion();

inline float roundf(float value)
{
    // Workaround Valgrind's mismatches: when running from Valgrind the call to std::round(4.5) == 4.0 instead of 5.0
    return (value < 0.f) ? ::floorf(value - 0.5f) : ::floorf(value + 0.5f);
}

} // namespace armnn
