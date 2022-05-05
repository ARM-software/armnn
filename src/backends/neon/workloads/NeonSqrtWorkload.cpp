//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonSqrtWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{

arm_compute::Status NeonSqrtWorkloadValidate(const TensorInfo& input, const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Sqrt;
    const arm_compute::ActivationLayerInfo activationLayerInfo =
            ConvertActivationDescriptorToAclActivationLayerInfo(descriptor);

    return arm_compute::NEActivationLayer::validate(&aclInput, &aclOutput, activationLayerInfo);
}

NeonSqrtWorkload::NeonSqrtWorkload(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info)
    : NeonBaseWorkload<ElementwiseUnaryQueueDescriptor>(descriptor, info)
{
    ARMNN_ASSERT(descriptor.m_Parameters.m_Operation == UnaryOperation::Sqrt);

    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonSqrtWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NeonSqrtWorkload", 1, 1);

    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = ActivationFunction::Sqrt;
    const arm_compute::ActivationLayerInfo activationLayerInfo =
            ConvertActivationDescriptorToAclActivationLayerInfo(activationDescriptor);

    arm_compute::ITensor& input  = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_SqrtLayer.configure(&input, &output, activationLayerInfo);
}

void NeonSqrtWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonSqrtWorkload_Execute", this->GetGuid());
    m_SqrtLayer.run();
}

} // namespace armnn