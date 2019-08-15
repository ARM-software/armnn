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
    DeviceSpec(const BackendIdSet& supportedBackends)
        : m_SupportedBackends{supportedBackends} {}

    virtual ~DeviceSpec() {}

    virtual const BackendIdSet& GetSupportedBackends() const override
    {
        return m_SupportedBackends;
    }

    void AddSupportedBackends(const BackendIdSet& backendIds)
    {
        m_SupportedBackends.insert(backendIds.begin(), backendIds.end());
    }

private:
    DeviceSpec() = delete;
    BackendIdSet m_SupportedBackends;
};

} // namespace armnn
