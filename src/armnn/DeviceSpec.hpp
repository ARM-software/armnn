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

    virtual const BackendIdSet& GetSupportedBackends() const
    {
        return m_SupportedBackends;
    }

private:
    DeviceSpec() = delete;
    BackendIdSet m_SupportedBackends;
};

}
