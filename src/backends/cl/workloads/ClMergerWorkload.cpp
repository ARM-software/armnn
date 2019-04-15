//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ClMergerWorkload.hpp"
#include "ClWorkloadUtils.hpp"
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <cl/ClTensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>

#include <arm_compute/core/Types.h>

#include <boost/polymorphic_pointer_cast.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

namespace
{
size_t CalcAxis(const MergerDescriptor& desc)
{
    return (desc.GetNumDimensions() - desc.GetConcatAxis()) - 1;
}
} //namespace

arm_compute::Status ClMergerWorkloadValidate(const std::vector<const TensorInfo*>& inputs,
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
    return arm_compute::CLConcatenateLayer::validate(aclInputPtrs, &aclOutputInfo, aclAxis);
}

ClMergerWorkload::ClMergerWorkload(const MergerQueueDescriptor& descriptor, const WorkloadInfo& info)
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

    std::vector<arm_compute::ICLTensor *> aclInputs;
    for (auto input : m_Data.m_Inputs)
    {
        arm_compute::ICLTensor& aclInput  = boost::polymorphic_pointer_downcast<IClTensorHandle>(input)->GetTensor();
        aclInputs.emplace_back(&aclInput);
    }
    arm_compute::ICLTensor& output = boost::polymorphic_pointer_downcast<IClTensorHandle>(
                                                                         m_Data.m_Outputs[0])->GetTensor();

    // Create the layer function
    m_Layer.reset(new arm_compute::CLConcatenateLayer());

    // Configure input and output tensors
    size_t aclAxis = CalcAxis(descriptor.m_Parameters);
    m_Layer->configure(aclInputs, &output, aclAxis);

    // Prepare
    m_Layer->prepare();
}

void ClMergerWorkload::Execute() const
{
    if (m_Layer)
    {
        ARMNN_SCOPED_PROFILING_EVENT_CL("ClMergerWorkload_Execute");
        m_Layer->run();
    }
}

} //namespace armnn