//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefFillWorkload.hpp"
#include "Fill.hpp"

#include "Decoders.hpp"
#include "Encoders.hpp"
#include "RefWorkloadUtils.hpp"
#include "Profiling.hpp"

namespace armnn
{

void RefFillWorkload::Execute() const
{
    Execute(m_Data.m_Outputs);
}

void RefFillWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Outputs);
}

void RefFillWorkload::Execute(std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefFillWorkload_Execute");

    const TensorInfo &outputTensorInfo = GetTensorInfo(outputs[0]);

    std::unique_ptr<Encoder<float>> encoderPtr = MakeEncoder<float>(outputTensorInfo, outputs[0]->Map());
    Encoder<float> &encoder = *encoderPtr;

    Fill(encoder, outputTensorInfo.GetShape(), m_Data.m_Parameters.m_Value);
}

} //namespace armnn
