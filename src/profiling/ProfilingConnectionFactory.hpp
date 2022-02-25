//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IProfilingConnectionFactory.hpp"

namespace arm
{

namespace pipe
{

class ProfilingConnectionFactory final : public IProfilingConnectionFactory
{
public:
    ProfilingConnectionFactory()  = default;
    ~ProfilingConnectionFactory() = default;

    IProfilingConnectionPtr GetProfilingConnection(const ProfilingOptions& options) const override;
};

}    // namespace pipe

}    // namespace arm
