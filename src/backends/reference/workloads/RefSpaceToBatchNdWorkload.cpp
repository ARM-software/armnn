//
// Copyright © 2018-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefSpaceToBatchNdWorkload.hpp"
#include "SpaceToBatchNd.hpp"

#include "RefWorkloadUtils.hpp"

namespace armnn
{

void RefSpaceToBatchNdWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefSpaceToBatchNdWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefSpaceToBatchNdWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefSpaceToBatchNdWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(inputInfo, inputs[0]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(outputInfo, outputs[0]->Map());

    SpaceToBatchNd(inputInfo, outputInfo, m_Data.m_Parameters, *inputDecoder, *outputEncoder);
}

} //namespace armnn
