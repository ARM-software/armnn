//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonSplitterWorkload.hpp"

#include <arm_compute/runtime/NEON/functions/NESplit.h>

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <neon/NeonTensorHandle.hpp>

#include "NeonWorkloadUtils.hpp"

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

arm_compute::Status NeonSplitterWorkloadValidate(const TensorInfo& input,
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
    return arm_compute::NESplit::validate(&aclInputInfo, aclOutputPtr, aclAxis);
}

NeonSplitterWorkload::NeonSplitterWorkload(const SplitterQueueDescriptor& descriptor, const WorkloadInfo& info)
    : NeonBaseWorkload<SplitterQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonSplitterWorkload_Construct",
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

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();

    std::vector<arm_compute::ITensor *> aclOutputs;
    for (auto output : m_Data.m_Outputs)
    {
        arm_compute::ITensor& aclOutput  = PolymorphicPointerDowncast<IAclTensorHandle>(output)->GetTensor();
        aclOutputs.emplace_back(&aclOutput);
    }

    // Create the layer function
    std::unique_ptr<arm_compute::NESplit> layer(new arm_compute::NESplit());

    // Configure input and output tensors
    std::set<unsigned int> splitAxis = ComputeSplitAxis(descriptor.m_Parameters, m_Data.m_Inputs[0]->GetShape());
    if (splitAxis.size() != 1)
    {
        throw InvalidArgumentException("Cannot derive split axis from SplitterDescriptor");
    }

    unsigned int aclAxis = CalcAclAxis(descriptor.m_Parameters.GetNumDimensions(), *splitAxis.begin());
    layer->configure(&input, aclOutputs, aclAxis);

    // Prepare
    layer->prepare();
    m_Layer.reset(layer.release());
}

void NeonSplitterWorkload::Execute() const
{
    if (m_Layer)
    {
        ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonSplitterWorkload_Execute", this->GetGuid());
        m_Layer->run();
    }
}

} //namespace armnn

