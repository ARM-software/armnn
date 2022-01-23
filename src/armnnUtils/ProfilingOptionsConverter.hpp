//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/IRuntime.hpp>
#include <armnn/profiling/ProfilingOptions.hpp>

namespace armnn
{

profiling::ProfilingOptions ConvertExternalProfilingOptions(
    const IRuntime::CreationOptions::ExternalProfilingOptions& options);

} // namespace armnn
