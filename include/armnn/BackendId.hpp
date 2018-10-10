//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <set>
#include <unordered_set>
#include <string>
#include <memory>

namespace armnn
{

//
// The Compute enum is now deprecated and it is now
// replaced by BackendId
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

inline std::ostream& operator<<(std::ostream& os, const std::vector<Compute>& compute)
{
    for (const Compute& comp : compute) {
        os << GetComputeDeviceAsCString(comp) << " ";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const std::set<Compute>& compute)
{
    for (const Compute& comp : compute) {
        os << GetComputeDeviceAsCString(comp) << " ";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Compute& compute)
{
    os << GetComputeDeviceAsCString(compute);
    return os;
}

class BackendId final
{
public:
    BackendId(const std::string& id) : m_Id{id} {}
    BackendId(const char* id) : m_Id{id} {}
    BackendId(Compute compute) : m_Id{GetComputeDeviceAsCString(compute)} {}

    operator std::string() const { return m_Id; }

    BackendId& operator=(const std::string& other)
    {
        m_Id = other;
        return *this;
    }

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

    bool operator<(const BackendId& other) const
    {
        return m_Id < other.m_Id;
    }

    const std::string& Get() const { return m_Id; }

private:
    // backend Id mustn't be empty:
    BackendId() = delete;
    std::string m_Id;
};

template <template <class...> class TContainer>
inline std::ostream& operator<<(std::ostream& os,
                                const TContainer<BackendId>& ids)
{
    os << '[';
    for (const auto& id : ids) { os << id.Get() << " "; }
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
