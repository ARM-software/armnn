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
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefActivationWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    Activation(*MakeDecoder<float>(inputInfo, m_Data.m_Inputs[0]->Map()),
               *MakeEncoder<float>(outputInfo, m_Data.m_Outputs[0]->Map()),
               inputInfo,
               m_Data.m_Parameters.m_Function,
               m_Data.m_Parameters.m_A,
               m_Data.m_Parameters.m_B);
}

} //namespace armnn
