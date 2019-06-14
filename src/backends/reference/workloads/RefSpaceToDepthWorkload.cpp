//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefSpaceToDepthWorkload.hpp"
#include "SpaceToDepth.hpp"

#include "RefWorkloadUtils.hpp"
#include <ResolveType.hpp>

namespace armnn
{

void RefSpaceToDepthWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefSpaceToDepthWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    std::unique_ptr<Decoder<float>> decoder = MakeDecoder<float>(inputInfo, m_Data.m_Inputs[0]->Map());

    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);
    std::unique_ptr<Encoder<float>> encoder = MakeEncoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());

    SpaceToDepth(inputInfo, outputInfo, m_Data.m_Parameters, *decoder, *encoder);
}

} //namespace armnn
