//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonMaximumWorkload.hpp"
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>

namespace armnn
{

arm_compute::Status NeonMaximumWorkloadValidate(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput0 = armcomputetensorutils::BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1 = armcomputetensorutils::BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::NEElementwiseMax::validate(&aclInput0,
                                                   &aclInput1,
                                                   &aclOutput);
}

NeonMaximumWorkload::NeonMaximumWorkload(const MaximumQueueDescriptor& descriptor,
                                         const WorkloadInfo& info)
    : NeonBaseWorkload<MaximumQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonMaximumWorkload", 2, 1);

    arm_compute::ITensor& input0 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& input1 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_MaxLayer.configure(&input0, &input1, &output);
}

void NeonMaximumWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonMaximumWorkload_Execute", this->GetGuid());
    m_MaxLayer.run();
}

} //namespace armnn
