//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonGatherWorkload.hpp"
#include "NeonWorkloadUtils.hpp"
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

namespace armnn
{
arm_compute::Status NeonGatherWorkloadValidate(const TensorInfo& input,
                                               const TensorInfo& indices,
                                               const TensorInfo& output,
                                               const GatherDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInput   = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclIndices = BuildArmComputeTensorInfo(indices);
    const arm_compute::TensorInfo aclOutput  = BuildArmComputeTensorInfo(output);

    int aclAxis = ComputeAclAxis(descriptor.m_Axis, input);

    return arm_compute::NEGather::validate(&aclInput, &aclIndices, &aclOutput, aclAxis);
}

NeonGatherWorkload::NeonGatherWorkload(const GatherQueueDescriptor& descriptor,
                                       const WorkloadInfo& info)
        : NeonBaseWorkload<GatherQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonGatherWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NeonGatherWorkload", 2, 1);

    arm_compute::ITensor& input   = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& indices = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output  = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    int aclAxis = ComputeAclAxis(descriptor.m_Parameters.m_Axis, info.m_InputTensorInfos[0]);

    m_Layer.configure(&input, &indices, &output, aclAxis);
}

void NeonGatherWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonGatherWorkload_Execute", this->GetGuid());
    m_Layer.run();
}
} //namespace armnn