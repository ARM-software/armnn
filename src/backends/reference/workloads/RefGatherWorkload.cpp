//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
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
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefGatherWorkload_Execute");

    const TensorInfo& inputInfo0 = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(m_Data.m_Inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    std::unique_ptr<Decoder<float>> decoderPtr = MakeDecoder<float>(inputInfo0, m_Data.m_Inputs[0]->Map());
    Decoder<float>& decoder = *decoderPtr;

    const int32_t* indicesData = GetInputTensorData<int32_t>(1, m_Data);

    std::unique_ptr<Encoder<float>> encoderPtr = MakeEncoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());
    Encoder<float>& encoder = *encoderPtr;

    Gather(inputInfo0, inputInfo1, outputInfo, decoder, indicesData, encoder, m_Data.m_Parameters.m_Axis);
}

} //namespace armnn
