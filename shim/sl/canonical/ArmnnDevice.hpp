//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
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
    ArmnnDevice(DriverOptions options);
    ~ArmnnDevice() {}
protected:
    armnn::IRuntimePtr m_Runtime;
    armnn::IGpuAccTunedParametersPtr m_ClTunedParameters;
    DriverOptions m_Options;
};

} // namespace armnn_driver
