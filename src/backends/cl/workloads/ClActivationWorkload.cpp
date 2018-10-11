//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClActivationWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <backends/CpuTensorHandle.hpp>
#include <backends/cl/ClLayerSupport.hpp>
#include <backends/cl/ClTensorHandle.hpp>
#include <backends/aclCommon/ArmComputeUtils.hpp>

namespace armnn
{
arm_compute::Status ClActivationWorkloadValidate(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const ActivationDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInput = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    const arm_compute::ActivationLayerInfo activationLayerInfo =
        ConvertActivationDescriptorToAclActivationLayerInfo(descriptor);

    if (input.GetDataType() == DataType::QuantisedAsymm8 &&
        activationLayerInfo.activation() == arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC)
    {
        return arm_compute::Status{arm_compute::ErrorCode::RUNTIME_ERROR,
                                   "CL: Logistic Activations unsupported with QAsymm8 data type."};
    }

    return arm_compute::CLActivationLayer::validate(&aclInput,
                                                    &aclOutput,
                                                    activationLayerInfo);
}

ClActivationWorkload::ClActivationWorkload(const ActivationQueueDescriptor& descriptor,
                                           const WorkloadInfo& info)
    : BaseWorkload<ActivationQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClActivationWorkload", 1, 1);

    const arm_compute::ActivationLayerInfo activationLayerInfo =
        ConvertActivationDescriptorToAclActivationLayerInfo(m_Data.m_Parameters);

    arm_compute::ICLTensor& input  = static_cast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<ClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    m_ActivationLayer.configure(&input, &output, activationLayerInfo);
}

void ClActivationWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClActivationWorkload_Execute");
    m_ActivationLayer.run();
}

} //namespace armnn
