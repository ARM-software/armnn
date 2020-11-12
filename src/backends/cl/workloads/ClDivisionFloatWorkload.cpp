//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClDivisionFloatWorkload.hpp"

#include <aclCommon/ArmComputeUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>

#include <cl/ClTensorHandle.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{

arm_compute::Status ClDivisionWorkloadValidate(const TensorInfo& input0,
                                               const TensorInfo& input1,
                                               const TensorInfo& output,
                                               const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInput1 = armcomputetensorutils::BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput2 = armcomputetensorutils::BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
            activationDescriptor);

    return arm_compute::CLArithmeticDivision::validate(&aclInput1, &aclInput2, &aclOutput, activationInfo);
}


ClDivisionFloatWorkload::ClDivisionFloatWorkload(const DivisionQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info)
    : FloatWorkload<DivisionQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClDivisionFloatWorkload", 2, 1);

    arm_compute::ICLTensor& input0 = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& input1 = static_cast<IClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    m_ArithmeticDivision.configure(&input0, &input1, &output, activationInfo);
}

void ClDivisionFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClDivisionFloatWorkload_Execute");
    RunClFunction(m_ArithmeticDivision, CHECK_LOCATION());
}

} //namespace armnn
