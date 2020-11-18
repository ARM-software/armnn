//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClLogicalNotWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <cl/ClTensorHandle.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClLogicalNotWorkloadValidate(const TensorInfo& input,
                                                 const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInputInfo  = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::Status aclStatus = arm_compute::CLLogicalNot::validate(&aclInputInfo,
                                                                              &aclOutputInfo);
    return aclStatus;
}

ClLogicalNotWorkload::ClLogicalNotWorkload(const ElementwiseUnaryQueueDescriptor& descriptor,
                                           const WorkloadInfo& info)
    : BaseWorkload<ElementwiseUnaryQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClLogicalNotWorkload", 1, 1);

    arm_compute::ICLTensor& input  = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_LogicalNotLayer.configure(&input, &output);
}

void ClLogicalNotWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClLogicalNotWorkload_Execute");
    m_LogicalNotLayer.run();
}

} // namespace armnn
