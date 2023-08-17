//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonFusedWorkload.hpp"
#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>

#include <arm_compute/runtime/NEON/functions/NEAddMulAdd.h>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status NeonFusedWorkloadValidate(const std::vector<std::reference_wrapper<TensorInfo>>& inputInfos,
                                              const std::vector<std::reference_wrapper<TensorInfo>>& outputInfos,
                                              const FusedDescriptor& fusedDescriptor,
                                              const ActivationDescriptor* activationDescriptor)
{
    std::vector<arm_compute::TensorInfo> actInputInfos;
    actInputInfos.reserve(inputInfos.size());
    for (size_t i = 0u; i < inputInfos.size(); ++i)
    {
        actInputInfos.emplace_back(BuildArmComputeTensorInfo(inputInfos[i]));
    }

    std::vector<arm_compute::TensorInfo> actOutputInfos;
    actOutputInfos.reserve(outputInfos.size());
    for (size_t i = 0u; i < outputInfos.size(); ++i)
    {
        actOutputInfos.emplace_back(BuildArmComputeTensorInfo(outputInfos[i]));
    }

    const arm_compute::ActivationLayerInfo activationInfo =
            ConvertActivationDescriptorToAclActivationLayerInfo(activationDescriptor);

    switch (fusedDescriptor.m_FusedKernelType)
    {
        case FusedKernelType::AddMulAdd:
            return arm_compute::NEAddMulAdd::validate(
                                &actInputInfos[0],
                                &actInputInfos[1],
                                &actInputInfos[2],  // bn_mul
                                &actInputInfos[3],  // bn_add
                                actOutputInfos.size() == 1 ? nullptr : &actOutputInfos[0], // add_output
                                actOutputInfos.size() == 1 ? &actOutputInfos[0] : &actOutputInfos[1], // final_output
                                arm_compute::ConvertPolicy::SATURATE,
                                activationInfo);
        default:
            return arm_compute::Status{arm_compute::ErrorCode::RUNTIME_ERROR,
                                       "NeonFusedWorkloadValidate: no valid kernel type"};
    }
}


NeonFusedWorkload::NeonFusedWorkload(const FusedQueueDescriptor& descriptor, const WorkloadInfo& info)
    : NeonBaseWorkload<FusedQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonFusedWorkload",
                                 static_cast<unsigned int>(info.m_InputTensorInfos.size()),
                                 static_cast<unsigned int>(info.m_OutputTensorInfos.size()));

    std::vector<arm_compute::ITensor*> inputs;
    inputs.reserve(info.m_InputTensorInfos.size());
    for (auto input : m_Data.m_Inputs)
    {
        inputs.emplace_back(&PolymorphicDowncast<IAclTensorHandle*>(input)->GetTensor());
    }

    std::vector<arm_compute::ITensor*> outputs;
    outputs.reserve(info.m_OutputTensorInfos.size());
    for (auto output : m_Data.m_Outputs)
    {
        outputs.emplace_back(&PolymorphicDowncast<IAclTensorHandle*>(output)->GetTensor());
    }

    const arm_compute::ActivationLayerInfo activationInfo =
            ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    switch (descriptor.m_Parameters.m_FusedKernelType)
    {
        case FusedKernelType::AddMulAdd:
        {
            auto layer = std::make_unique<arm_compute::NEAddMulAdd>();
            layer->configure(inputs[0],
                             inputs[1],
                             inputs[2],  // bn_mul
                             inputs[3],  // bn_add
                             outputs.size() == 1 ? nullptr : outputs[0], // add_output
                             outputs.size() == 1 ? outputs[0] : outputs[1], // final_output
                             arm_compute::ConvertPolicy::SATURATE,
                             activationInfo);
            m_FusedLayer.reset(layer.release());
            break;
        }
        default:
            throw Exception("NeonFusedWorkload: no valid kernel type.");
    }
}

void NeonFusedWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonFusedWorkload_Execute", this->GetGuid());
    m_FusedLayer->run();
}

} //namespace armnn

