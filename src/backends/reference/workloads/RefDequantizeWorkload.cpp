//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefDequantizeWorkload.hpp"
#include "RefWorkloadUtils.hpp"
#include "Encoders.hpp"
#include "Decoders.hpp"
#include "Dequantize.hpp"

namespace armnn
{

void RefDequantizeWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefDequantizeWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefDequantizeWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDequantizeWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    auto inputDecoder  = MakeDecoder<float>(inputInfo,  inputs[0]->Map());
    auto outputEncoder = MakeEncoder<float>(outputInfo, outputs[0]->Map());

    Dequantize(*inputDecoder, *outputEncoder, inputInfo, outputInfo);
}

} // namespace armnn
