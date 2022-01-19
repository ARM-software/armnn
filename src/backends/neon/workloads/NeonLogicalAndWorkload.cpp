//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonLogicalAndWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{

arm_compute::Status NeonLogicalAndWorkloadValidate(const TensorInfo& input0,
                                                   const TensorInfo& input1,
                                                   const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInputInfo0 = BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInputInfo1 = BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::Status aclStatus = arm_compute::NELogicalAnd::validate(&aclInputInfo0,
                                                                              &aclInputInfo1,
                                                                              &aclOutputInfo);
    return aclStatus;
}

NeonLogicalAndWorkload::NeonLogicalAndWorkload(const LogicalBinaryQueueDescriptor& descriptor,
                                               const WorkloadInfo& info)
    : NeonBaseWorkload<LogicalBinaryQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonLogicalAndWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NeonLogicalAndWorkload", 2, 1);

    arm_compute::ITensor& input0 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& input1 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_LogicalAndLayer.configure(&input0, &input1, &output);
}

void NeonLogicalAndWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonLogicalAndWorkload_Execute", this->GetGuid());
    m_LogicalAndLayer.run();
}

} // namespace armnn
