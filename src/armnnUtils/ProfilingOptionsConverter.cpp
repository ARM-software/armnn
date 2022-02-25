//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingOptionsConverter.hpp"

#include <algorithm>
#include <iterator>

namespace arm
{

namespace pipe
{

ProfilingOptions ConvertExternalProfilingOptions(
    const armnn::IRuntime::CreationOptions::ExternalProfilingOptions& options)
{
    ProfilingOptions convertedOptions;
    convertedOptions.m_EnableProfiling     = options.m_EnableProfiling;
    convertedOptions.m_TimelineEnabled     = options.m_TimelineEnabled;
    convertedOptions.m_OutgoingCaptureFile = options.m_OutgoingCaptureFile;
    convertedOptions.m_IncomingCaptureFile = options.m_IncomingCaptureFile;
    convertedOptions.m_FileOnly            = options.m_FileOnly;
    convertedOptions.m_CapturePeriod       = options.m_CapturePeriod;
    convertedOptions.m_FileFormat          = options.m_FileFormat;
    std::copy(options.m_LocalPacketHandlers.begin(), options.m_LocalPacketHandlers.end(),
              std::back_inserter(convertedOptions.m_LocalPacketHandlers));
    return convertedOptions;
}

} // namespace arm

} // namespace pipe
