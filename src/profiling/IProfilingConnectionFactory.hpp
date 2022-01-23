//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IProfilingConnection.hpp"

#include <armnn/profiling/ProfilingOptions.hpp>

#include <memory>

namespace armnn
{

namespace profiling
{

class IProfilingConnectionFactory
{
public:
    using ExternalProfilingOptions = ProfilingOptions;
    using IProfilingConnectionPtr = std::unique_ptr<IProfilingConnection>;

    virtual ~IProfilingConnectionFactory() {}

    virtual IProfilingConnectionPtr GetProfilingConnection(const ProfilingOptions& options) const = 0;
};

} // namespace profiling

} // namespace armnn
