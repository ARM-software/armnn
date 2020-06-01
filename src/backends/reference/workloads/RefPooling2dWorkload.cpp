//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefPooling2dWorkload.hpp"

#include "Pooling2d.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"
#include "BaseIterator.hpp"

namespace armnn
{
void RefPooling2dWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefPooling2dWorkload_Execute");

    const TensorInfo& inputInfo  = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    auto inputDecoder  = MakeDecoder<float>(inputInfo,  m_Data.m_Inputs[0] ->Map());
    auto outputEncoder = MakeEncoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());

    Pooling2d(*inputDecoder,
              *outputEncoder,
              inputInfo,
              outputInfo,
              m_Data.m_Parameters);
}
} //namespace armnn
