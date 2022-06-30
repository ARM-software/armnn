//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefSoftmaxWorkload.hpp"

#include "Decoders.hpp"
#include "Encoders.hpp"
#include "RefWorkloadUtils.hpp"
#include "Softmax.hpp"

#include "Profiling.hpp"

#include <vector>

namespace armnn
{

void RefSoftmaxWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefSoftmaxWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefSoftmaxWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefSoftmaxWorkload_Execute");

    const TensorInfo &inputTensorInfo = GetTensorInfo(inputs[0]);

    std::unique_ptr<Decoder<float>> decoderPtr = MakeDecoder<float>(inputTensorInfo, inputs[0]->Map());
    Decoder<float> &decoder = *decoderPtr;

    const TensorInfo &outputTensorInfo = GetTensorInfo(outputs[0]);

    std::unique_ptr<Encoder<float>> encoderPtr = MakeEncoder<float>(outputTensorInfo, outputs[0]->Map());
    Encoder<float> &encoder = *encoderPtr;

    Softmax(decoder,
            encoder,
            inputTensorInfo,
            m_Data.m_Parameters.m_Beta,
            m_Data.m_Parameters.m_Axis);
}
} //namespace armnn
