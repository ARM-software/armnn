//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/BackendId.hpp>
#include <vector>

namespace armnn
{

struct BackendSettings
{
    BackendIdVector m_PreferredBackends;
    BackendIdSet    m_SupportedBackends;
    BackendIdSet    m_SelectedBackends;
    BackendIdSet    m_IgnoredBackends;

    BackendSettings() = default;

    BackendSettings(const BackendIdVector& preferredBackends,
                    const IDeviceSpec& deviceSpec)
    {
        Initialize(preferredBackends, deviceSpec);
    }

    bool IsBackendPreferred(const BackendId& backend) const
    {
        return IsBackendInCollection(backend, m_PreferredBackends);
    }

    bool IsBackendSupported(const BackendId& backend) const
    {
        return IsBackendInCollection(backend, m_SupportedBackends);
    }

    bool IsBackendSelected(const BackendId& backend) const
    {
        return IsBackendInCollection(backend, m_SelectedBackends);
    }

    bool IsBackendIgnored(const BackendId& backend) const
    {
        return IsBackendInCollection(backend, m_IgnoredBackends);
    }

    bool IsCpuRefUsed() const
    {
        BackendId cpuBackendId(Compute::CpuRef);
        return IsBackendSupported(cpuBackendId) && IsBackendPreferred(cpuBackendId);
    }

    BackendIdVector GetAvailablePreferredBackends() const
    {
        BackendIdVector availablePreferredBackends;
        for (const BackendId& backend : m_PreferredBackends)
        {
            if (IsBackendSupported(backend) && !IsBackendIgnored(backend))
            {
                availablePreferredBackends.push_back(backend);
            }
        }
        return availablePreferredBackends;
    }

private:
    void Initialize(const BackendIdVector& preferredBackends,
                    const IDeviceSpec& deviceSpec)
    {
        // Copy preferred backends from input
        m_PreferredBackends = preferredBackends;

        // Obtain list of supported backends
        const DeviceSpec& spec = *boost::polymorphic_downcast<const DeviceSpec*>(&deviceSpec);
        m_SupportedBackends = spec.GetSupportedBackends();
    }

    template<typename Collection>
    bool IsBackendInCollection(const BackendId& backend, const Collection& collection) const
    {
        return std::find(collection.begin(), collection.end(), backend) != collection.end();
    }
};

} //namespace armnn
