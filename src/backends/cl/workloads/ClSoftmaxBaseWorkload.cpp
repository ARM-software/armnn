//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClSoftmaxBaseWorkload.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/runtime/CL/functions/CLSoftmaxLayer.h>

namespace armnn
{

arm_compute::Status ClSoftmaxWorkloadValidate(const TensorInfo& input,
                                              const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::CLSoftmaxLayer::validate(&aclInputInfo, &aclOutputInfo);
}

}
