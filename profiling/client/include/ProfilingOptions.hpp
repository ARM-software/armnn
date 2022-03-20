//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ILocalPacketHandler.hpp"

#include <string>
#include <vector>

namespace arm
{
namespace pipe
{
/// The lowest performance data capture interval we support is 10 miliseconds.
constexpr unsigned int LOWEST_CAPTURE_PERIOD = 10000u;

struct ProfilingOptions {
    ProfilingOptions()
    : m_EnableProfiling(false), m_TimelineEnabled(false), m_OutgoingCaptureFile(""),
      m_IncomingCaptureFile(""), m_FileOnly(false), m_CapturePeriod(arm::pipe::LOWEST_CAPTURE_PERIOD),
      m_FileFormat("binary"), m_LocalPacketHandlers() {}

    /// Indicates whether external profiling is enabled or not.
    bool m_EnableProfiling;
    /// Indicates whether external timeline profiling is enabled or not.
    bool m_TimelineEnabled;
    /// Path to a file in which outgoing timeline profiling messages will be stored.
    std::string m_OutgoingCaptureFile;
    /// Path to a file in which incoming timeline profiling messages will be stored.
    std::string m_IncomingCaptureFile;
    /// Enable profiling output to file only.
    bool m_FileOnly;
    /// The duration at which captured profiling messages will be flushed.
    uint32_t m_CapturePeriod;
    /// The format of the file used for outputting profiling data.
    std::string m_FileFormat;
    std::vector <ILocalPacketHandlerSharedPtr> m_LocalPacketHandlers;
};

} // namespace pipe

} // namespace arm
