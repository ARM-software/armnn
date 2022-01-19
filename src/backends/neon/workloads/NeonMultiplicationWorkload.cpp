//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonMultiplicationWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeUtils.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h>

namespace armnn
{

arm_compute::Status NeonMultiplicationWorkloadValidate(const TensorInfo& input0,
                                                       const TensorInfo& input1,
                                                       const TensorInfo& output,
                                                       const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInput1 = armcomputetensorutils::BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput2 = armcomputetensorutils::BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    auto convertPolicy = (IsQuantizedType(input0.GetDataType()) || IsQuantizedType(input1.GetDataType())) ?
                          arm_compute::ConvertPolicy::SATURATE :
                          arm_compute::ConvertPolicy::WRAP;

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
            activationDescriptor);

    // At the time of writing, configure() will fail if a rounding policy other than TO_ZERO is supplied to it,
    // when providing a scale of 1.0 for F32 tensors, even though the provided rounding policy appears to be
    // ignored for F32 tensors.
    return arm_compute::NEPixelWiseMultiplication::validate(&aclInput1,
                                                            &aclInput2,
                                                            &aclOutput,
                                                            1.0f,
                                                            convertPolicy,
                                                            arm_compute::RoundingPolicy::TO_ZERO,
                                                            activationInfo);
}

NeonMultiplicationWorkload::NeonMultiplicationWorkload(const MultiplicationQueueDescriptor& descriptor,
                                                       const WorkloadInfo& info)
    : NeonBaseWorkload<MultiplicationQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonMultiplicationWorkload", 2, 1);

    arm_compute::ITensor& input1 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& input2 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    auto convertPolicy = (IsQuantizedType(info.m_InputTensorInfos[0].GetDataType()) ||
                          IsQuantizedType(info.m_InputTensorInfos[1].GetDataType())) ?
                          arm_compute::ConvertPolicy::SATURATE :
                          arm_compute::ConvertPolicy::WRAP;

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    // At the time of writing, configure() will fail if a rounding policy other than TO_ZERO is supplied to it,
    // when providing a scale of 1.0 for F32 tensors, even though the provided rounding policy appears to be
    // ignored for F32 tensors.
    auto layer = std::make_unique<arm_compute::NEPixelWiseMultiplication>();
    layer->configure(&input1,
                     &input2,
                     &output,
                     1.0f,
                     convertPolicy,
                     arm_compute::RoundingPolicy::TO_ZERO,
                     activationInfo);
    m_PixelWiseMultiplication.reset(layer.release());
}

void NeonMultiplicationWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonMultiplicationWorkload_Execute", this->GetGuid());
    m_PixelWiseMultiplication->run();
}

} //namespace armnn
