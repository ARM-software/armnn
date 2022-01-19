//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonLogicalNotWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>


namespace armnn
{

arm_compute::Status NeonLogicalNotWorkloadValidate(const TensorInfo& input,
                                                   const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInputInfo  = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::Status aclStatus = arm_compute::NELogicalNot::validate(&aclInputInfo,
                                                                              &aclOutputInfo);
    return aclStatus;
}

NeonLogicalNotWorkload::NeonLogicalNotWorkload(const ElementwiseUnaryQueueDescriptor& descriptor,
                                               const WorkloadInfo& info)
    : NeonBaseWorkload<ElementwiseUnaryQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonLogicalNotWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NeonLogicalNotWorkload", 1, 1);

    arm_compute::ITensor& input  = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_LogicalNotLayer.configure(&input, &output);
}

void NeonLogicalNotWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonLogicalNotWorkload_Execute", this->GetGuid());
    m_LogicalNotLayer.run();
}

} // namespace armnn
