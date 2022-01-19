//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonRsqrtWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>


namespace armnn
{

arm_compute::Status NeonRsqrtWorkloadValidate(const TensorInfo& input, const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::NERsqrtLayer::validate(&aclInput, &aclOutput);
}

NeonRsqrtWorkload::NeonRsqrtWorkload(const RsqrtQueueDescriptor& descriptor, const WorkloadInfo& info)
    : NeonBaseWorkload<RsqrtQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonRsqrtWorkload", 1, 1);

    arm_compute::ITensor& input  = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_RsqrtLayer.configure(&input, &output);
}

void NeonRsqrtWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonRsqrtWorkload_Execute", this->GetGuid());
    m_RsqrtLayer.run();
}

} // namespace armnn
