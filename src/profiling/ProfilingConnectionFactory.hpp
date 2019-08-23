//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IProfilingConnection.hpp"

#include <Runtime.hpp>

#include <memory>

namespace armnn
{

namespace profiling
{

class ProfilingConnectionFactory final
{
public:
    ProfilingConnectionFactory()  = default;
    ~ProfilingConnectionFactory() = default;

    std::unique_ptr<IProfilingConnection> GetProfilingConnection(
        const Runtime::CreationOptions::ExternalProfilingOptions& options) const;
};

} // namespace profiling

} // namespace armnn
