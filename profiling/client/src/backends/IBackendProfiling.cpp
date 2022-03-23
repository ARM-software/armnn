//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BackendProfiling.hpp"

#include <client/include/backends/IBackendProfiling.hpp>

namespace arm
{

namespace pipe
{

std::unique_ptr<IBackendProfiling> IBackendProfiling::CreateBackendProfiling(const ProfilingOptions& options,
                                                                             IProfilingService& profilingService,
                                                                             const std::string& backendId)
{
    return std::make_unique<BackendProfiling>(options, profilingService, backendId);
}

} // namespace pipe
} // namespace arm
