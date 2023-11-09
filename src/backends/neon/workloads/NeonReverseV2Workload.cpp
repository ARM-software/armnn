//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonReverseV2Workload.hpp"
#include "NeonWorkloadUtils.hpp"
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <backendsCommon/WorkloadUtils.hpp>

namespace armnn
{
arm_compute::Status NeonReverseV2WorkloadValidate(const TensorInfo& input,
                                                  const TensorInfo& axis,
                                                  const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclAxis = BuildArmComputeTensorInfo(axis);
    const arm_compute::TensorInfo aclOutput = BuildArmComputeTensorInfo(output);

    return arm_compute::NEReverse::validate(&aclInput, &aclOutput, &aclAxis, true);
}

NeonReverseV2Workload::NeonReverseV2Workload(const ReverseV2QueueDescriptor& descriptor,
                                             const WorkloadInfo& info)
        : BaseWorkload<ReverseV2QueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonReverseV2Workload", 2, 1);

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& axis = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_Layer.configure(&input, &output, &axis, true);
}

void NeonReverseV2Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_NAME_GUID("NeonReverseV2Workload_Execute");
    m_Layer.run();
}

} // namespace armnn