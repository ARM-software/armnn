//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonAdditionWorkload.hpp"
#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>

#include <arm_compute/runtime/NEON/functions/NEArithmeticAddition.h>

namespace armnn
{

arm_compute::Status NeonAdditionWorkloadValidate(const TensorInfo& input0,
                                                 const TensorInfo& input1,
                                                 const TensorInfo& output,
                                                 const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInput0 = armcomputetensorutils::BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1 = armcomputetensorutils::BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
            activationDescriptor);

    return arm_compute::NEArithmeticAddition::validate(&aclInput0,
                                                       &aclInput1,
                                                       &aclOutput,
                                                       arm_compute::ConvertPolicy::SATURATE,
                                                       activationInfo);
}


NeonAdditionWorkload::NeonAdditionWorkload(const AdditionQueueDescriptor& descriptor,
                                           const WorkloadInfo& info)
    : NeonBaseWorkload<AdditionQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonAdditionWorkload", 2, 1);

    arm_compute::ITensor& input1 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& input2 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    auto layer = std::make_unique<arm_compute::NEArithmeticAddition>();
    layer->configure(&input1, &input2, &output, arm_compute::ConvertPolicy::SATURATE, activationInfo);
    m_AddLayer.reset(layer.release());
}

void NeonAdditionWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonAdditionWorkload_Execute", this->GetGuid());
    m_AddLayer->run();
}

} //namespace armnn

