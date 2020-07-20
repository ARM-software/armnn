//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IProfilingConnectionFactory.hpp"

namespace armnn
{

namespace profiling
{

class ProfilingConnectionFactory final : public IProfilingConnectionFactory
{
public:
    ProfilingConnectionFactory()  = default;
    ~ProfilingConnectionFactory() = default;

    IProfilingConnectionPtr GetProfilingConnection(const ExternalProfilingOptions& options) const override;
};

}    // namespace profiling

}    // namespace armnn
