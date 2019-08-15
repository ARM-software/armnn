//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClSoftmaxBaseWorkload.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include <arm_compute/runtime/CL/functions/CLSoftmaxLayer.h>

namespace armnn
{

arm_compute::Status ClSoftmaxWorkloadValidate(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const SoftmaxDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    unsigned int aclAxis = ComputeSoftmaxAclAxis(descriptor, input);
    return arm_compute::CLSoftmaxLayer::validate(&aclInputInfo, &aclOutputInfo, descriptor.m_Beta, aclAxis);
}

}
