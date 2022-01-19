//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClAdditionWorkload.hpp"

#include <cl/ClTensorHandle.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

static constexpr arm_compute::ConvertPolicy g_AclConvertPolicy = arm_compute::ConvertPolicy::SATURATE;

ClAdditionWorkload::ClAdditionWorkload(const AdditionQueueDescriptor& descriptor,
                                       const WorkloadInfo& info,
                                       const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<AdditionQueueDescriptor>(descriptor, info)
{
    this->m_Data.ValidateInputsOutputs("ClAdditionWorkload", 2, 1);

    arm_compute::ICLTensor& input0 = static_cast<IClTensorHandle*>(this->m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& input1 = static_cast<IClTensorHandle*>(this->m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(this->m_Data.m_Outputs[0])->GetTensor();

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClAdditionWorkload_configure");
        m_Layer.configure(clCompileContext, &input0, &input1, &output, g_AclConvertPolicy, activationInfo);
    }
}

void ClAdditionWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClAdditionWorkload_Execute", this->GetGuid());
    RunClFunction(m_Layer, CHECK_LOCATION());
}

arm_compute::Status ClAdditionValidate(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output,
                                       const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInput0Info = BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1Info = BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
            activationDescriptor);

    const arm_compute::Status aclStatus = arm_compute::CLArithmeticAddition::validate(&aclInput0Info,
                                                                                      &aclInput1Info,
                                                                                      &aclOutputInfo,
                                                                                      g_AclConvertPolicy,
                                                                                      activationInfo);

    return aclStatus;
}

} //namespace armnn
