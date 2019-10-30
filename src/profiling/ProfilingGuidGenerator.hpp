//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IProfilingGuidGenerator.hpp"

#include <functional>

namespace armnn
{

namespace profiling
{

class ProfilingGuidGenerator : public IProfilingGuidGenerator
{
public:
    /// Construct a generator with the default address space static/dynamic partitioning
    ProfilingGuidGenerator() : m_Sequence(0) {}

    /// Return the next random Guid in the sequence
    // NOTE: dummy implementation for the moment
    inline ProfilingDynamicGuid NextGuid() override
    {
        // NOTE: skipping the zero for testing purposes
        ProfilingDynamicGuid guid(++m_Sequence);
        return guid;
    }

    /// Create a ProfilingStaticGuid based on a hash of the string
    // NOTE: dummy implementation for the moment
    inline ProfilingStaticGuid GenerateStaticId(const std::string& str) override
    {
        uint64_t guid = static_cast<uint64_t>(m_StringHasher(str));
        return guid;
    }

private:
    std::hash<std::string> m_StringHasher;
    uint64_t m_Sequence;
};

} // namespace profiling

} // namespace armnn
