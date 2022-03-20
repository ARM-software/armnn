//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/IRuntime.hpp>

#include <client/include/ProfilingOptions.hpp>

namespace arm
{

namespace pipe
{

ProfilingOptions ConvertExternalProfilingOptions(
    const armnn::IRuntime::CreationOptions::ExternalProfilingOptions& options);

} // namespace pipe

} // namespace arm
