//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonMergerWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <neon/NeonTensorHandle.hpp>



namespace armnn
{
using namespace armcomputetensorutils;

namespace
{
size_t CalcAxis(const armnn::MergerDescriptor& desc)
{
    return (desc.GetNumDimensions() - desc.GetConcatAxis()) - 1;
}
} //namespace

arm_compute::Status NeonMergerWorkloadValidate(const std::vector<const TensorInfo*>& inputs,
                                               const TensorInfo& output,
                                               const MergerDescriptor& descriptor)

{
    std::vector<arm_compute::TensorInfo> aclInputs;
    for (const TensorInfo* input : inputs)
    {
        arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(*input, armnn::DataLayout::NCHW);
        aclInputs.emplace_back(aclInputInfo);
    }
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);
    std::vector<arm_compute::ITensorInfo*> aclInputPtrs;
    for (arm_compute::ITensorInfo& input : aclInputs)
    {
        aclInputPtrs.emplace_back(&input);
    }

    size_t aclAxis = CalcAxis(descriptor);
    return arm_compute::NEConcatenateLayer::validate(aclInputPtrs, &aclOutputInfo, aclAxis);
}

NeonMergerWorkload::NeonMergerWorkload(
const MergerQueueDescriptor& descriptor, const WorkloadInfo& info)
        : BaseWorkload<MergerQueueDescriptor>(descriptor, info)
{
    bool allInputsAreSubtensors = true;

    // Check that all inputs are sub-tensors
    for (auto input : descriptor.m_Inputs)
    {
        if (!input->GetParent())
        {
            // Non sub-tensor input found so we need to execute the merger function
            allInputsAreSubtensors = false;
            break;
        }
    }

    if (allInputsAreSubtensors)
    {
        // Can skip configuring the merger function since it's not executed
        return;
    }

    std::vector<arm_compute::ITensor *> aclInputs;
    for (auto input : m_Data.m_Inputs)
    {
        arm_compute::ITensor& aclInput  = boost::polymorphic_pointer_downcast<INeonTensorHandle>(input)->GetTensor();
        aclInputs.emplace_back(&aclInput);
    }
    arm_compute::ITensor& output = boost::polymorphic_pointer_downcast<INeonTensorHandle>(
        m_Data.m_Outputs[0])->GetTensor();

    // Create the layer function
    m_Layer.reset(new arm_compute::NEConcatenateLayer());

    // Configure input and output tensors
    size_t aclAxis = CalcAxis(descriptor.m_Parameters);
    m_Layer->configure(aclInputs, &output, aclAxis);

    // Prepare
    m_Layer->prepare();
}

void NeonMergerWorkload::Execute() const
{
    if (m_Layer)
    {
        ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonMergerWorkload_Execute");
        m_Layer->run();
    }
}

} //namespace armnn

