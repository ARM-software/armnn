//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <set>
#include <vector>

namespace armnn
{

class DeviceSpec : public IDeviceSpec
{
public:
    DeviceSpec()
    {}

    DeviceSpec(const BackendIdSet& supportedBackends)
        : m_SupportedBackends{supportedBackends} {}

    virtual ~DeviceSpec() {}

    virtual const BackendIdSet& GetSupportedBackends() const override
    {
        return m_SupportedBackends;
    }

    void AddSupportedBackends(const BackendIdSet& backendIds, bool isDynamic = false)
    {
        m_SupportedBackends.insert(backendIds.begin(), backendIds.end());
        if (isDynamic)
        {
            m_DynamicBackends.insert(backendIds.begin(), backendIds.end());
        }
    }

    void ClearDynamicBackends()
    {
        for (const auto& id : m_DynamicBackends)
        {
            m_SupportedBackends.erase(id);
        }
        m_DynamicBackends.clear();
    }

    const BackendIdSet& GetDynamicBackends() const
    {
        return m_DynamicBackends;
    }

private:
    BackendIdSet m_SupportedBackends;
    BackendIdSet m_DynamicBackends;
};

} // namespace armnn
