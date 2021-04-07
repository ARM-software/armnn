//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefActivationWorkload.hpp"

#include "Activation.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefActivationWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefActivationWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefActivationWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefActivationWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    Activation(*MakeDecoder<float>(inputInfo, inputs[0]->Map()),
               *MakeEncoder<float>(outputInfo, outputs[0]->Map()),
               inputInfo,
               m_Data.m_Parameters.m_Function,
               m_Data.m_Parameters.m_A,
               m_Data.m_Parameters.m_B);
}


} //namespace armnn
