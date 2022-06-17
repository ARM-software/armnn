//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "canonical/ArmnnDriver.hpp"

#include <nnapi/IDevice.h>

namespace android::nn
{

std::vector<SharedDevice> getDevices()
{
    return { std::make_shared<armnn_driver::ArmnnDriver>(DriverOptions()) };
}

}  // namespace android::nn
