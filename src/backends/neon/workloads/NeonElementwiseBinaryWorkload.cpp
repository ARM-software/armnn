//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonElementwiseBinaryWorkload.hpp"
#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>

#include <arm_compute/runtime/NEON/functions/NEElementwiseOperations.h>

namespace armnn
{

arm_compute::Status NeonElementwiseBinaryWorkloadValidate(const TensorInfo& input0,
                                                          const TensorInfo& input1,
                                                          const TensorInfo& output,
                                                          const ElementwiseBinaryDescriptor& descriptor,
                                                          const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInput0 = armcomputetensorutils::BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1 = armcomputetensorutils::BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
            activationDescriptor);

    switch (descriptor.m_Operation)
    {
        case armnn::BinaryOperation::Power:
            return arm_compute::NEElementwisePower::validate(&aclInput0,
                                                             &aclInput1,
                                                             &aclOutput,
                                                             activationInfo);
        case armnn::BinaryOperation::SqDiff:
            return arm_compute::NEElementwiseSquaredDiff::validate(&aclInput0,
                                                                   &aclInput1,
                                                                   &aclOutput,
                                                                   activationInfo);
        default:
            throw InvalidArgumentException("Unknown binary operator", CHECK_LOCATION());
    }
}


NeonElementwiseBinaryWorkload::NeonElementwiseBinaryWorkload(const ElementwiseBinaryQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info)
    : NeonBaseWorkload<ElementwiseBinaryQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonElementwiseBinaryWorkload", 2, 1);

    arm_compute::ITensor& input1 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& input2 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "NeonElementwiseBinaryWorkload_configure");

    switch (descriptor.m_Parameters.m_Operation)
    {
        case armnn::BinaryOperation::Power:
        {
            auto powerLayer = std::make_unique<arm_compute::NEElementwisePower>();
            powerLayer->configure(&input1, &input2, &output, activationInfo);
            m_ElementwiseBinaryLayer.reset(powerLayer.release());
            break;
        }
        case armnn::BinaryOperation::SqDiff:
        {
            auto SqDiffLayer = std::make_unique<arm_compute::NEElementwiseSquaredDiff>();
            SqDiffLayer->configure(&input1, &input2, &output, activationInfo);
            m_ElementwiseBinaryLayer.reset(SqDiffLayer.release());
            break;
        }
        default:
            throw InvalidArgumentException("Unknown binary operator", CHECK_LOCATION());
    }
}

void NeonElementwiseBinaryWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_NAME_GUID("NeonElementwiseBinaryWorkload_Execute");
    m_ElementwiseBinaryLayer->run();
}

} //namespace armnn