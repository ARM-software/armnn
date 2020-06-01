//
// Copyright © 2017 Arm Ltd. All rights reserved.
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
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefSoftmaxWorkload_Execute");

    const TensorInfo &inputTensorInfo = GetTensorInfo(m_Data.m_Inputs[0]);

    std::unique_ptr<Decoder<float>> decoderPtr = MakeDecoder<float>(inputTensorInfo, m_Data.m_Inputs[0]->Map());
    Decoder<float> &decoder = *decoderPtr;

    const TensorInfo &outputTensorInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    std::unique_ptr<Encoder<float>> encoderPtr = MakeEncoder<float>(outputTensorInfo, m_Data.m_Outputs[0]->Map());
    Encoder<float> &encoder = *encoderPtr;

    Softmax(decoder,
            encoder,
            inputTensorInfo,
            m_Data.m_Parameters.m_Beta,
            m_Data.m_Parameters.m_Axis);
}
} //namespace armnn
