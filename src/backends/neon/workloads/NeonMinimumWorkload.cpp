//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonMinimumWorkload.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>

namespace armnn
{

arm_compute::Status NeonMinimumWorkloadValidate(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput0 = armcomputetensorutils::BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1 = armcomputetensorutils::BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::NEElementwiseMin::validate(&aclInput0,
                                                   &aclInput1,
                                                   &aclOutput);
}

NeonMinimumWorkload::NeonMinimumWorkload(const MinimumQueueDescriptor& descriptor,
                                         const WorkloadInfo& info)
    : NeonBaseWorkload<MinimumQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonMinimumWorkload", 2, 1);

    arm_compute::ITensor& input0 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& input1 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_MinLayer.configure(&input0, &input1, &output);
}

void NeonMinimumWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonMinimumWorkload_Execute", this->GetGuid());
    m_MinLayer.run();
}

} //namespace armnn
