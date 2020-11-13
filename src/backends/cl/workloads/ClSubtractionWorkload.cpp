//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClSubtractionWorkload.hpp"

#include <cl/ClTensorHandle.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

static constexpr arm_compute::ConvertPolicy g_AclConvertPolicy = arm_compute::ConvertPolicy::SATURATE;

ClSubtractionWorkload::ClSubtractionWorkload(const SubtractionQueueDescriptor& descriptor,
                                             const WorkloadInfo& info)
    : BaseWorkload<SubtractionQueueDescriptor>(descriptor, info)
{
    this->m_Data.ValidateInputsOutputs("ClSubtractionWorkload", 2, 1);

    arm_compute::ICLTensor& input0 = static_cast<IClTensorHandle*>(this->m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& input1 = static_cast<IClTensorHandle*>(this->m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(this->m_Data.m_Outputs[0])->GetTensor();

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    m_Layer.configure(&input0, &input1, &output, g_AclConvertPolicy, activationInfo);
}

void ClSubtractionWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClSubtractionWorkload_Execute");
    RunClFunction(m_Layer, CHECK_LOCATION());
}

arm_compute::Status ClSubtractionValidate(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInput0Info = BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInput1Info = BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
            activationDescriptor);

    const arm_compute::Status aclStatus = arm_compute::CLArithmeticSubtraction::validate(&aclInput0Info,
                                                                                         &aclInput1Info,
                                                                                         &aclOutputInfo,
                                                                                         g_AclConvertPolicy,
                                                                                         activationInfo);

    return aclStatus;
}

} //namespace armnn
