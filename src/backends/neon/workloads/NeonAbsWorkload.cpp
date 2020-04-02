//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonAbsWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{

arm_compute::Status NeonAbsWorkloadValidate(const TensorInfo& input, const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::NEAbsLayer::validate(&aclInput, &aclOutput);
}

NeonAbsWorkload::NeonAbsWorkload(const AbsQueueDescriptor& descriptor, const WorkloadInfo& info)
    : BaseWorkload<AbsQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonAbsWorkload", 1, 1);

    arm_compute::ITensor& input  = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_AbsLayer.configure(&input, &output);
}

void NeonAbsWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonAbsWorkload_Execute");
    m_AbsLayer.run();
}

} // namespace armnn
