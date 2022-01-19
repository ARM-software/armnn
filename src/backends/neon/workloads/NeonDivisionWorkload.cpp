//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonDivisionWorkload.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <armnn/backends/TensorHandle.hpp>

namespace armnn
{

arm_compute::Status NeonDivisionWorkloadValidate(const TensorInfo& input0,
                                                 const TensorInfo& input1,
                                                 const TensorInfo& output,
                                                 const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInput0 = armcomputetensorutils::BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1 = armcomputetensorutils::BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
            activationDescriptor);

    return arm_compute::NEElementwiseDivision::validate(&aclInput0,
                                                        &aclInput1,
                                                        &aclOutput,
                                                        activationInfo);
}

NeonDivisionWorkload::NeonDivisionWorkload(const DivisionQueueDescriptor& descriptor,
                                         const WorkloadInfo& info)
    : NeonBaseWorkload<DivisionQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonDivisionWorkload", 2, 1);

    arm_compute::ITensor& input0 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& input1 = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    m_DivLayer.configure(&input0, &input1, &output, activationInfo);
}

void NeonDivisionWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonDivisionWorkload_Execute", this->GetGuid());
    m_DivLayer.run();
}

} //namespace armnn
