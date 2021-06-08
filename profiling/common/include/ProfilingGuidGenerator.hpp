//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IProfilingGuidGenerator.hpp"
#include "ProfilingGuid.hpp"

#include <functional>
#include <mutex>

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
    inline ProfilingDynamicGuid NextGuid() override
    {
        std::lock_guard<std::mutex> sequencelock(m_SequenceMutex);
        ProfilingDynamicGuid guid(m_Sequence);
        m_Sequence++;
        if (m_Sequence >= MIN_STATIC_GUID)
        {
            // Reset the sequence to 0 when it reaches the upper bound of dynamic guid
            m_Sequence = 0;
        }
        return guid;
    }

    /// Create a ProfilingStaticGuid based on a hash of the string
    inline ProfilingStaticGuid GenerateStaticId(const std::string& str) override
    {
        uint64_t staticHash = m_Hash(str) | MIN_STATIC_GUID;
        return ProfilingStaticGuid(staticHash);
    }

    /// Reset the generator back to zero. Used mainly for test.
    inline void Reset()
    {
        std::lock_guard<std::mutex> sequencelock(m_SequenceMutex);
        m_Sequence = 0;
    }

private:
    std::hash<std::string> m_Hash;
    uint64_t m_Sequence;
    std::mutex m_SequenceMutex;
};

} // namespace profiling

} // namespace armnn
