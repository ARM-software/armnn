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

    virtual std::vector<IBackendPtr> GetBackends() const
    {
        return std::vector<IBackendPtr>();
    }

    std::set<Compute> m_SupportedComputeDevices;
};

}
