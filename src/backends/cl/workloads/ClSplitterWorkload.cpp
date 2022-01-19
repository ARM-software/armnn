//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClSplitterWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <cl/ClTensorHandle.hpp>


namespace armnn
{

using namespace armcomputetensorutils;

namespace
{
    unsigned int CalcAclAxis(unsigned int numDimensions, unsigned int splitAxis)
    {
        return (numDimensions - splitAxis) - 1;
    }

} //namespace

arm_compute::Status ClSplitterWorkloadValidate(const TensorInfo& input,
                                               const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                                               unsigned int splitAxis)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input);

    size_t numOutputs = outputs.size();

    std::vector<arm_compute::TensorInfo> aclOutputs;
    aclOutputs.reserve(numOutputs);

    std::vector<arm_compute::ITensorInfo*> aclOutputPtr;
    aclOutputPtr.reserve(numOutputs);

    for (size_t i = 0u; i < outputs.size(); ++i)
    {
        aclOutputs.emplace_back(BuildArmComputeTensorInfo(outputs[i]));
        aclOutputPtr.emplace_back(&aclOutputs.back());
    }

    unsigned int aclAxis = CalcAclAxis(input.GetNumDimensions(), splitAxis);
    return arm_compute::CLSplit::validate(&aclInputInfo, aclOutputPtr, aclAxis);
}

ClSplitterWorkload::ClSplitterWorkload(const SplitterQueueDescriptor& descriptor,
                                       const WorkloadInfo& info,
                                       const arm_compute::CLCompileContext&)
        : ClBaseWorkload<SplitterQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClSplitterWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());
    bool allOutputsAreSubtensors = true;

    // Check that all outputs are sub-tensors
    for (auto output : m_Data.m_Outputs)
    {
        if (output && !output->GetParent())
        {
            // Non sub-tensor input found so we need to execute the split function
            allOutputsAreSubtensors = false;
            break;
        }
    }

    if (allOutputsAreSubtensors)
    {
        // Can skip configuring the split function since it's not executed
        return;
    }

    arm_compute::ICLTensor& input = armnn::PolymorphicPointerDowncast<IClTensorHandle>(
            m_Data.m_Inputs[0])->GetTensor();

    std::vector<arm_compute::ICLTensor *> aclOutputs;
    for (auto output : m_Data.m_Outputs)
    {
        arm_compute::ICLTensor& aclOutput  = armnn::PolymorphicPointerDowncast<IClTensorHandle>(output)->GetTensor();
        aclOutputs.emplace_back(&aclOutput);
    }

    // Create the layer function

    // Configure input and output tensors
    std::set<unsigned int> splitAxis = ComputeSplitAxis(descriptor.m_Parameters, m_Data.m_Inputs[0]->GetShape());
    if (splitAxis.size() != 1)
    {
        throw InvalidArgumentException("Cannot derive split axis from SplitterDescriptor");
    }

    unsigned int aclAxis = CalcAclAxis(descriptor.m_Parameters.GetNumDimensions(), *splitAxis.begin());
    auto layer = std::make_unique<arm_compute::CLSplit>();
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClSplitterWorkload_configure");
        layer->configure(&input, aclOutputs, aclAxis);
    }

    // Prepare
    layer->prepare();

    m_Layer = std::move(layer);
}

void ClSplitterWorkload::Execute() const
{
    if (m_Layer)
    {
        ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClSplitterWorkload_Execute", this->GetGuid());
        m_Layer->run();
    }
}

} //namespace armnn
