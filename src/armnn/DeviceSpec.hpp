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
    DeviceSpec() {}
    virtual ~DeviceSpec() {}

    virtual std::vector<IBackendSharedPtr> GetBackends() const
    {
        return std::vector<IBackendSharedPtr>();
    }

    std::set<Compute> m_SupportedComputeDevices;
};

}
