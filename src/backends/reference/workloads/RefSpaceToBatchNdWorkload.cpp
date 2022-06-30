//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefSpaceToBatchNdWorkload.hpp"
#include "SpaceToBatchNd.hpp"

#include "RefWorkloadUtils.hpp"
#include <ResolveType.hpp>

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
    std::unique_ptr<Decoder<float>> decoder = MakeDecoder<float>(inputInfo, inputs[0]->Map());

    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);
    std::unique_ptr<Encoder<float>> encoder = MakeEncoder<float>(outputInfo, outputs[0]->Map());

    SpaceToBatchNd(inputInfo, outputInfo, m_Data.m_Parameters, *decoder, *encoder);
}

} //namespace armnn
