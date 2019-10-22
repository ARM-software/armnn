//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <stdint.h>

namespace armnn
{

namespace profiling
{

class ProfilingGuid
{
public:
    ProfilingGuid(uint64_t guid) : m_Guid(guid) {}

    operator uint64_t() const { return m_Guid; }

    bool operator==(const ProfilingGuid& other) const
    {
        return m_Guid == other.m_Guid;
    }

    bool operator!=(const ProfilingGuid& other) const
    {
        return m_Guid != other.m_Guid;
    }

    bool operator<(const ProfilingGuid& other) const
    {
        return m_Guid < other.m_Guid;
    }

    bool operator<=(const ProfilingGuid& other) const
    {
        return m_Guid <= other.m_Guid;
    }

    bool operator>(const ProfilingGuid& other) const
    {
        return m_Guid > other.m_Guid;
    }

    bool operator>=(const ProfilingGuid& other) const
    {
        return m_Guid >= other.m_Guid;
    }

protected:
    uint64_t m_Guid;
};

/// Strongly typed guids to distinguish between those generated at runtime, and those that are statically defined.
struct ProfilingDynamicGuid : public ProfilingGuid
{
    using ProfilingGuid::ProfilingGuid;
};

struct ProfilingStaticGuid : public ProfilingGuid
{
    using ProfilingGuid::ProfilingGuid;
};

} // namespace profiling

} // namespace armnn
