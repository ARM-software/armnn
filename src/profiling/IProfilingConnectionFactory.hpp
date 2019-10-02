//
// Copyright © 2019 Arm Ltd. All rights reserved.
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

class IProfilingConnectionFactory
{
public:
    using ExternalProfilingOptions = Runtime::CreationOptions::ExternalProfilingOptions;
    using IProfilingConnectionPtr = std::unique_ptr<IProfilingConnection>;

    virtual ~IProfilingConnectionFactory() {}

    virtual IProfilingConnectionPtr GetProfilingConnection(const ExternalProfilingOptions& options) const = 0;
};

} // namespace profiling

} // namespace armnn
