//
// Copyright © 2017 Arm Ltd. All rights reserved.
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
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefFloorFloat32Workload_Execute");

    const TensorInfo &inputTensorInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    std::unique_ptr<Decoder<float>> decoderPtr = MakeDecoder<float>(inputTensorInfo, m_Data.m_Inputs[0]->Map());
    Decoder<float> &decoder = *decoderPtr;

    const TensorInfo &outputTensorInfo = GetTensorInfo(m_Data.m_Outputs[0]);
    std::unique_ptr<Encoder<float>> encoderPtr = MakeEncoder<float>(outputTensorInfo, m_Data.m_Outputs[0]->Map());
    Encoder<float> &encoder = *encoderPtr;

    unsigned int numElements = GetTensorInfo(m_Data.m_Inputs[0]).GetNumElements();

    for (unsigned int i = 0; i < numElements; ++i)
    {
        encoder.Set(floorf(decoder.Get()));
        ++decoder;
        ++encoder;
    }
}

} //namespace armnn
