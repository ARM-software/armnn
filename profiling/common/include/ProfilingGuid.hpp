//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <memory>
#include <stdint.h>

namespace armnn
{

namespace profiling
{

static constexpr uint64_t MIN_STATIC_GUID = 1llu << 63;

class ProfilingGuid
{
public:
    ProfilingGuid() : m_Guid(0) {}

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



namespace std
{
/// make ProfilingGuid hashable
template <>
struct hash<armnn::profiling::ProfilingGuid>
{
    std::size_t operator()(armnn::profiling::ProfilingGuid const& guid) const noexcept
    {
        return hash<uint64_t>()(uint64_t(guid));
    }
};

/// make ProfilingDynamicGuid hashable
template <>
struct hash<armnn::profiling::ProfilingDynamicGuid>
{
    std::size_t operator()(armnn::profiling::ProfilingDynamicGuid const& guid) const noexcept
    {
        return hash<uint64_t>()(uint64_t(guid));
    }
};

/// make ProfilingStaticGuid hashable
template <>
struct hash<armnn::profiling::ProfilingStaticGuid>
{
    std::size_t operator()(armnn::profiling::ProfilingStaticGuid const& guid) const noexcept
    {
        return hash<uint64_t>()(uint64_t(guid));
    }
};

}  // namespace std