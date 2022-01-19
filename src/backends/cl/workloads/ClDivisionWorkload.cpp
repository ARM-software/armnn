//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClDivisionWorkload.hpp"

#include <aclCommon/ArmComputeUtils.hpp>
#include <armnn/backends/TensorHandle.hpp>

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


ClDivisionWorkload::ClDivisionWorkload(const DivisionQueueDescriptor& descriptor,
                                       const WorkloadInfo& info,
                                       const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<DivisionQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClDivisionWorkload", 2, 1);

    arm_compute::ICLTensor& input0 = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& input1 = static_cast<IClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClDivisionWorkload_configure");
        m_ArithmeticDivision.configure(clCompileContext, &input0, &input1, &output, activationInfo);
    }
}

void ClDivisionWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClDivisionWorkload_Execute", this->GetGuid());
    RunClFunction(m_ArithmeticDivision, CHECK_LOCATION());
}

} //namespace armnn
