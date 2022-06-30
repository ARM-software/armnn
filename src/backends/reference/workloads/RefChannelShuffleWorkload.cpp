//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/backends/ITensorHandleFactory.hpp>
#include <armnnUtils/Transpose.hpp>
#include "RefChannelShuffleWorkload.hpp"
#include "RefWorkloadUtils.hpp"
#include "Profiling.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"

namespace armnn
{
void RefChannelShuffleWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefChannelShuffleWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

// Reference implementation for channel shuffle taken from
// https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/master/nn/common/operations/ChannelShuffle.cpp
void RefChannelShuffleWorkload::Execute(std::vector<ITensorHandle*> inputs,
                                        std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefChannelShuffleWorkload_Execute");

    const TensorInfo& inputInfo  = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);
    std::unique_ptr<Decoder<float>> decoderPtr = MakeDecoder<float>(inputInfo, inputs[0]->Map());
    Decoder<float>& decoder = *decoderPtr;

    std::unique_ptr<Encoder<float>> encoderPtr = MakeEncoder<float>(outputInfo, outputs[0]->Map());
    Encoder<float>& encoder = *encoderPtr;

    auto getNumberOfElements = [](const TensorShape& tensorShape,uint32_t startAxis, uint32_t lastAxis)
    {
        uint32_t count = 1;
        for (uint32_t i = startAxis; i < lastAxis; i++)
        {
            count *= tensorShape[i];
        }
        return count;
    };
    const TensorShape tensorShape = GetTensorInfo(inputs[0]).GetShape();
    uint32_t channelsAxis = m_Data.m_Parameters.m_Axis; // channelsAxis to perform channel shuffle on

    const uint32_t numGroups = m_Data.m_Parameters.m_NumGroups;
    const uint32_t groupSize = tensorShape[channelsAxis] / numGroups;

    uint32_t outerSize = getNumberOfElements(tensorShape, 0, channelsAxis);
    uint32_t innerSize = getNumberOfElements(tensorShape, channelsAxis + 1, tensorShape.GetNumDimensions());

    for (uint32_t outer = 0; outer < outerSize; ++outer)
    {
        for (uint32_t inner = 0; inner < innerSize; ++inner)
        {
            uint32_t decoderStep1 = outer * tensorShape[channelsAxis] * innerSize + inner;
            decoder += decoderStep1;
            uint32_t encoderStep1 = outer * tensorShape[channelsAxis] * innerSize + inner;
            encoder += encoderStep1;
            for (uint32_t i = 0; i < groupSize; i++)
            {
                for (uint32_t j = 0; j < numGroups; j++, encoder += innerSize, encoderStep1 += innerSize)
                {
                    decoder += innerSize * (i + j * groupSize);
                    float decoded = decoder.Get();
                    encoder.Set(decoded);
                    decoder -= innerSize * (i + j * groupSize);
                }
            }
            decoder -= decoderStep1;
            encoder -= encoderStep1;
        }
    }
}
}