//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefFloorWorkload.hpp"

#include "Decoders.hpp"
#include "Encoders.hpp"
#include "RefWorkloadUtils.hpp"
#include "Profiling.hpp"

namespace armnn
{

void RefFloorWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefFloorWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefFloorWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefFloorFloat32Workload_Execute");

    const TensorInfo &inputTensorInfo = GetTensorInfo(inputs[0]);
    std::unique_ptr<Decoder<float>> decoderPtr = MakeDecoder<float>(inputTensorInfo, inputs[0]->Map());
    Decoder<float> &decoder = *decoderPtr;

    const TensorInfo &outputTensorInfo = GetTensorInfo(outputs[0]);
    std::unique_ptr<Encoder<float>> encoderPtr = MakeEncoder<float>(outputTensorInfo, outputs[0]->Map());
    Encoder<float> &encoder = *encoderPtr;

    unsigned int numElements = GetTensorInfo(inputs[0]).GetNumElements();

    for (unsigned int i = 0; i < numElements; ++i)
    {
        encoder.Set(floorf(decoder.Get()));
        ++decoder;
        ++encoder;
    }
}

} //namespace armnn
