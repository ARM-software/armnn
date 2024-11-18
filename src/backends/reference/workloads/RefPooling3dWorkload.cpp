//
// Copyright © 2021-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefPooling3dWorkload.hpp"

#include "Pooling3d.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"
#include "BaseIterator.hpp"

namespace armnn
{
void RefPooling3dWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefPooling3dWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT_REF_NAME_GUID("RefPooling3dWorkload_Execute");

    const TensorInfo& inputInfo  = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    auto inputDecoder  = MakeDecoder<float>(inputInfo,  inputs[0] ->Map());
    auto outputEncoder = MakeEncoder<float>(outputInfo, outputs[0]->Map());

    Pooling3d(*inputDecoder,
              *outputEncoder,
              inputInfo,
              outputInfo,
              m_Data.m_Parameters);
}
} //namespace armnn
