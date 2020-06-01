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
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDequantizeWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    auto inputDecoder  = MakeDecoder<float>(inputInfo,  m_Data.m_Inputs[0]->Map());
    auto outputEncoder = MakeEncoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());

    Dequantize(*inputDecoder, *outputEncoder, inputInfo, outputInfo);
}

} // namespace armnn
