//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefGatherWorkload.hpp"

#include "Gather.hpp"
#include "Profiling.hpp"
#include "RefWorkloadUtils.hpp"
#include <ResolveType.hpp>

namespace armnn
{

void RefGatherWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefGatherWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefGatherWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefGatherWorkload_Execute");

    const TensorInfo& inputInfo0 = GetTensorInfo(inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    std::unique_ptr<Decoder<float>> decoderPtr = MakeDecoder<float>(inputInfo0, inputs[0]->Map());
    Decoder<float>& decoder = *decoderPtr;

    const int32_t* indicesData = reinterpret_cast<int32_t*>(inputs[1]->Map());

    std::unique_ptr<Encoder<float>> encoderPtr = MakeEncoder<float>(outputInfo, outputs[0]->Map());
    Encoder<float>& encoder = *encoderPtr;

    Gather(inputInfo0, inputInfo1, outputInfo, decoder, indicesData, encoder, m_Data.m_Parameters.m_Axis);
}

} //namespace armnn
