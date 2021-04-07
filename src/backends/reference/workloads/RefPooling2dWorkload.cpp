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
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefPooling2dWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefPooling2dWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefPooling2dWorkload_Execute");

    const TensorInfo& inputInfo  = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    auto inputDecoder  = MakeDecoder<float>(inputInfo,  inputs[0] ->Map());
    auto outputEncoder = MakeEncoder<float>(outputInfo, outputs[0]->Map());

    Pooling2d(*inputDecoder,
              *outputEncoder,
              inputInfo,
              outputInfo,
              m_Data.m_Parameters);
}
} //namespace armnn
