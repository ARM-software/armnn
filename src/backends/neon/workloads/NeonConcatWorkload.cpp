//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonConcatWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <neon/NeonTensorHandle.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

namespace
{
size_t CalcAxis(const armnn::OriginsDescriptor& descriptor)
{
    return (descriptor.GetNumDimensions() - descriptor.GetConcatAxis()) - 1;
}
} //namespace

arm_compute::Status NeonConcatWorkloadValidate(const std::vector<const TensorInfo*>& inputs,
                                               const TensorInfo& output,
                                               const OriginsDescriptor& descriptor)

{
    std::vector<arm_compute::TensorInfo> aclInputs;
    for (const TensorInfo* input : inputs)
    {
        arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(*input, armnn::DataLayout::NCHW);
        aclInputs.emplace_back(aclInputInfo);
    }
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);
    std::vector<const arm_compute::ITensorInfo*> aclInputPtrs;
    for (arm_compute::ITensorInfo& input : aclInputs)
    {
        aclInputPtrs.emplace_back(&input);
    }

    size_t aclAxis = CalcAxis(descriptor);
    return arm_compute::NEConcatenateLayer::validate(aclInputPtrs, &aclOutputInfo, aclAxis);
}

NeonConcatWorkload::NeonConcatWorkload(
const ConcatQueueDescriptor& descriptor, const WorkloadInfo& info)
        : NeonBaseWorkload<ConcatQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonConcatWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    bool allInputsAreSubtensors = true;

    // Check that all inputs are sub-tensors
    for (auto input : descriptor.m_Inputs)
    {
        if (!input->GetParent())
        {
            // Non sub-tensor input found so we need to execute the concat function
            allInputsAreSubtensors = false;
            break;
        }
    }

    if (allInputsAreSubtensors)
    {
        // Can skip configuring the concat function since it's not executed
        return;
    }

    std::vector<const arm_compute::ITensor *> aclInputs;
    for (auto input : m_Data.m_Inputs)
    {
        arm_compute::ITensor& aclInput  = armnn::PolymorphicPointerDowncast<IAclTensorHandle>(input)->GetTensor();
        aclInputs.emplace_back(&aclInput);
    }
    arm_compute::ITensor& output = armnn::PolymorphicPointerDowncast<IAclTensorHandle>(
        m_Data.m_Outputs[0])->GetTensor();

    // Create the layer function
    m_Layer.reset(new arm_compute::NEConcatenateLayer());

    // Configure input and output tensors
    size_t aclAxis = CalcAxis(descriptor.m_Parameters);
    m_Layer->configure(aclInputs, &output, aclAxis);

    // Prepare
    m_Layer->prepare();
}

void NeonConcatWorkload::Execute() const
{
    if (m_Layer)
    {
        ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonConcatWorkload_Execute", this->GetGuid());
        m_Layer->run();
    }
}

} //namespace armnn

