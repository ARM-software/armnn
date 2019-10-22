//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingGuid.hpp"

namespace armnn
{

namespace profiling
{

class ProfilingGuidGenerator
{
public:
    /// Construct a generator with the default address space static/dynamic partitioning
    ProfilingGuidGenerator() : m_Sequence(0) {}

    /// Return the next random Guid in the sequence
    ProfilingDynamicGuid NextGuid();

    /// Create a ProfilingStaticGuid based on a hash of the name
    ProfilingStaticGuid GenerateStaticId(const char* name);

private:
    uint64_t m_Sequence;
};

} // namespace profiling

} // namespace armnn
