//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace armnn
{

///
/// The Compute enum is now deprecated and it is now
/// being replaced by BackendId
///
enum class Compute
{
    Undefined = 0,
    /// CPU Execution: Reference C++ kernels
    CpuRef    = 1,
    /// CPU Execution: NEON: ArmCompute
    CpuAcc    = 2,
    /// GPU Execution: OpenCL: ArmCompute
    GpuAcc    = 3
};

/// Deprecated function that will be removed together with
/// the Compute enum
constexpr char const* GetComputeDeviceAsCString(Compute compute)
{
    switch (compute)
    {
        case armnn::Compute::CpuRef: return "CpuRef";
        case armnn::Compute::CpuAcc: return "CpuAcc";
        case armnn::Compute::GpuAcc: return "GpuAcc";
        default:                     return "Unknown";
    }
}

/// Deprecated function that will be removed together with
/// the Compute enum
inline std::ostream& operator<<(std::ostream& os, const std::vector<Compute>& compute)
{
    for (const Compute& comp : compute)
    {
        os << GetComputeDeviceAsCString(comp) << " ";
    }
    return os;
}

/// Deprecated function that will be removed together with
/// the Compute enum
inline std::ostream& operator<<(std::ostream& os, const std::set<Compute>& compute)
{
    for (const Compute& comp : compute)
    {
        os << GetComputeDeviceAsCString(comp) << " ";
    }
    return os;
}

/// Deprecated function that will be removed together with
/// the Compute enum
inline std::ostream& operator<<(std::ostream& os, const Compute& compute)
{
    os << GetComputeDeviceAsCString(compute);
    return os;
}

class BackendId final
{
public:
    BackendId() : m_Id(GetComputeDeviceAsCString(Compute::Undefined)) {}
    BackendId(const std::string& id) : m_Id{id} {}
    BackendId(const char* id) : m_Id{id} {}


    BackendId(const BackendId& other) = default;
    BackendId(BackendId&& other) = default;
    BackendId& operator=(const BackendId& other) = default;
    BackendId& operator=(BackendId&& other) = default;
    ~BackendId(){}

    /// Deprecated function that will be removed together with
    /// the Compute enum
    BackendId(Compute compute) : m_Id{GetComputeDeviceAsCString(compute)} {}

    operator std::string() const { return m_Id; }
    BackendId& operator=(const std::string& other)
    {
        m_Id = other;
        return *this;
    }

    /// Deprecated function that will be removed together with
    /// the Compute enum
    BackendId& operator=(Compute compute)
    {
        BackendId temp{compute};
        std::swap(temp.m_Id, m_Id);
        return *this;
    }

    bool operator==(const BackendId& other) const
    {
        return m_Id == other.m_Id;
    }

    /// comparison against objects from which the
    /// BackendId can be constructed
    template <typename O>
    bool operator==(const O& other) const
    {
        BackendId temp{other};
        return *this == temp;
    }

    template <typename O>
    bool operator!=(const O& other) const
    {
        return !(*this == other);
    }

    bool operator<(const BackendId& other) const
    {
        return m_Id < other.m_Id;
    }

    bool IsCpuRef() const { return m_Id == GetComputeDeviceAsCString(Compute::CpuRef); }

    const std::string& Get() const { return m_Id; }

    bool IsEmpty() const { return m_Id.empty(); }
    bool IsUndefined() const { return m_Id == GetComputeDeviceAsCString(Compute::Undefined); }

private:
    std::string m_Id;
};

} // namespace armnn

namespace std
{

/// make BackendId compatible with std hashtables by reusing the hash
/// function for strings.
/// Note this must come *before* the first use of unordered_set<BackendId>.
template <>
struct hash<armnn::BackendId>
{
    std::size_t operator()(const armnn::BackendId& id) const noexcept
    {
        std::hash<std::string> hasher;
        return hasher(id.Get());
    }
};

} // namespace std

namespace armnn
{

namespace profiling
{
    // Static constant describing ArmNN as a dummy backend
    static const BackendId BACKEND_ID("ARMNN");
} // profiling

inline std::ostream& operator<<(std::ostream& os, const BackendId& id)
{
    os << id.Get();
    return os;
}

template <template <typename...> class TContainer, typename... TContainerTemplateArgs>
std::ostream& operator<<(std::ostream& os,
                         const TContainer<BackendId, TContainerTemplateArgs...>& ids)
{
    os << '[';
    for (const auto& id : ids) { os << id << " "; }
    os << ']';
    return os;
}

using BackendIdVector = std::vector<BackendId>;
using BackendIdSet    = std::unordered_set<BackendId>;

} // namespace armnn

