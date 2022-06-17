//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DriverOptions.hpp"

#include <armnn/ArmNN.hpp>

namespace armnn_driver
{

class ArmnnDevice
{

protected:
    ArmnnDevice(DriverOptions options);
    virtual ~ArmnnDevice() {}

protected:
    armnn::IRuntimePtr m_Runtime;
    armnn::IGpuAccTunedParametersPtr m_ClTunedParameters;
    DriverOptions m_Options;
};

} // namespace armnn_driver
