//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClLogicalAndWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <cl/ClTensorHandle.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClLogicalAndWorkloadValidate(const TensorInfo& input0,
                                                 const TensorInfo& input1,
                                                 const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInputInfo0 = BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInputInfo1 = BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::Status aclStatus = arm_compute::CLLogicalAnd::validate(&aclInputInfo0,
                                                                              &aclInputInfo1,
                                                                              &aclOutputInfo);
    return aclStatus;
}

ClLogicalAndWorkload::ClLogicalAndWorkload(const LogicalBinaryQueueDescriptor& descriptor,
                                           const WorkloadInfo& info)
    : BaseWorkload<LogicalBinaryQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClLogicalAndWorkload", 2, 1);

    arm_compute::ICLTensor& input0 = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& input1 = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_LogicalAndLayer.configure(&input0, &input1, &output);
}

void ClLogicalAndWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClLogicalAndWorkload_Execute");
    m_LogicalAndLayer.run();
}

} // namespace armnn
