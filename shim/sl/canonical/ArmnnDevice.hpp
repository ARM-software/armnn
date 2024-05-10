//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DriverOptions.hpp"

#include <armnn/ArmNN.hpp>

namespace armnn_driver
{

class ArmnnDevice
{
friend class ArmnnDriver;

public:
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("The Shim and support library will be removed from Arm NN in 24.08", "24.08")
    ArmnnDevice(DriverOptions options);
    ~ArmnnDevice() {}
protected:
    armnn::IRuntimePtr m_Runtime;
    armnn::IGpuAccTunedParametersPtr m_ClTunedParameters;
    DriverOptions m_Options;
};

} // namespace armnn_driver
