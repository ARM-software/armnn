//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>

#include <string>

namespace armnn
{

namespace profiling
{

class IProfilingGuidGenerator
{
public:
    /// Return the next random Guid in the sequence
    virtual ProfilingDynamicGuid NextGuid() = 0;

    /// Create a ProfilingStaticGuid based on a hash of the string
    virtual ProfilingStaticGuid GenerateStaticId(const std::string& str) = 0;

    virtual ~IProfilingGuidGenerator() {}
};

} // namespace profiling

} // namespace armnn
