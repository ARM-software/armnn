//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ClConcatWorkload.hpp"
#include "ClWorkloadUtils.hpp"
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <cl/ClTensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>

#include <arm_compute/core/Types.h>

namespace armnn
{
using namespace armcomputetensorutils;

namespace
{
size_t CalcAxis(const OriginsDescriptor& descriptor)
{
    return (descriptor.GetNumDimensions() - descriptor.GetConcatAxis()) - 1;
}
} //namespace

arm_compute::Status ClConcatWorkloadValidate(const std::vector<const TensorInfo*>& inputs,
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
    return arm_compute::CLConcatenateLayer::validate(aclInputPtrs, &aclOutputInfo, aclAxis);
}

ClConcatWorkload::ClConcatWorkload(const ConcatQueueDescriptor& descriptor,
                                   const WorkloadInfo& info,
                                   const arm_compute::CLCompileContext& clCompileContext)
: ClBaseWorkload<ConcatQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClConcatWorkload_Construct",
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

    std::vector<const arm_compute::ICLTensor *> aclInputs;
    for (auto input : m_Data.m_Inputs)
    {
        arm_compute::ICLTensor& aclInput  = armnn::PolymorphicPointerDowncast<IClTensorHandle>(input)->GetTensor();
        aclInputs.emplace_back(&aclInput);
    }

    arm_compute::ICLTensor& output =
            armnn::PolymorphicPointerDowncast<IClTensorHandle>(m_Data.m_Outputs[0])->GetTensor();

    // Create the layer function
    auto layer = std::make_unique<arm_compute::CLConcatenateLayer>();

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClConcatWorkload_configure");
        // Configure input and output tensors
        size_t aclAxis = CalcAxis(descriptor.m_Parameters);
        layer->configure(clCompileContext, aclInputs, &output, aclAxis);
    }

    // Prepare
    layer->prepare();
    m_Layer = std::move(layer);
}

void ClConcatWorkload::Execute() const
{
    if (m_Layer)
    {
        ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClConcatWorkload_Execute", this->GetGuid());
        m_Layer->run();
    }
}

} //namespace armnn
