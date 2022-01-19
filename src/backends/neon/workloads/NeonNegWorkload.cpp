//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonNegWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{

arm_compute::Status NeonNegWorkloadValidate(const TensorInfo& input, const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::NENegLayer::validate(&aclInput, &aclOutput);
}

NeonNegWorkload::NeonNegWorkload(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info)
    : NeonBaseWorkload<ElementwiseUnaryQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonNegWorkload", 1, 1);

    arm_compute::ITensor& input  = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_NegLayer.configure(&input, &output);
}

void NeonNegWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonNegWorkload_Execute", this->GetGuid());
    m_NegLayer.run();
}

} // namespace armnn
