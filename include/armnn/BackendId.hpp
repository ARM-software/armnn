//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <ostream>
#include <set>
#include <unordered_set>
#include <string>
#include <memory>
#include <vector>

namespace armnn
{

//
// The Compute enum is now deprecated and it is now
// being replaced by BackendId
//
enum class Compute
{
    /// CPU Execution: Reference C++ kernels
    CpuRef      = 0,
    /// CPU Execution: NEON: ArmCompute
    CpuAcc      = 1,
    /// GPU Execution: OpenCL: ArmCompute
    GpuAcc      = 2,
    Undefined   = 5
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
    for (const Compute& comp : compute) {
        os << GetComputeDeviceAsCString(comp) << " ";
    }
    return os;
}

/// Deprecated function that will be removed together with
/// the Compute enum
inline std::ostream& operator<<(std::ostream& os, const std::set<Compute>& compute)
{
    for (const Compute& comp : compute) {
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

struct UninitializedBackendId {};

class BackendId final
{
public:
    BackendId() { GetComputeDeviceAsCString(Compute::Undefined); }
    BackendId(UninitializedBackendId) { GetComputeDeviceAsCString(Compute::Undefined); }
    BackendId(const std::string& id) : m_Id{id} {}
    BackendId(const char* id) : m_Id{id} {}

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

    // comparison against objects from which the
    // BackendId can be constructed
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

    const std::string& Get() const { return m_Id; }

private:
    std::string m_Id;
};

inline std::ostream& operator<<(std::ostream& os, const BackendId& id)
{
    os << id.Get();
    return os;
}

template <template <class...> class TContainer>
inline std::ostream& operator<<(std::ostream& os,
                                const TContainer<BackendId>& ids)
{
    os << '[';
    for (const auto& id : ids) { os << id << " "; }
    os << ']';
    return os;
}

using BackendIdSet = std::unordered_set<BackendId>;

} // namespace armnn

namespace std
{

// make BackendId compatible with std hashtables by reusing the hash
// function for strings
template <>
struct hash<armnn::BackendId>
{
    std::size_t operator()(const armnn::BackendId& id) const
    {
        std::hash<std::string> hasher;
        return hasher(id.Get());
    }
};

} // namespace std
