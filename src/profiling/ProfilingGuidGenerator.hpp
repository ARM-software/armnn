//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IProfilingGuidGenerator.hpp"

namespace armnn
{

namespace profiling
{

class ProfilingGuidGenerator : public IProfilingGuidGenerator
{
public:
    /// Construct a generator with the default address space static/dynamic partitioning
    ProfilingGuidGenerator() {}

    /// Return the next random Guid in the sequence
    // NOTE: dummy implementation for the moment
    inline ProfilingDynamicGuid NextGuid() override { return ProfilingDynamicGuid(0); }

    /// Create a ProfilingStaticGuid based on a hash of the string
    // NOTE: dummy implementation for the moment
    inline ProfilingStaticGuid GenerateStaticId(const std::string& str) override { return ProfilingStaticGuid(0); }
};

} // namespace profiling

} // namespace armnn
