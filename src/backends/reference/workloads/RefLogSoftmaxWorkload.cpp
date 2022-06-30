//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefLogSoftmaxWorkload.hpp"

#include "Decoders.hpp"
#include "Encoders.hpp"
#include "LogSoftmax.hpp"
#include "RefWorkloadUtils.hpp"

#include <Profiling.hpp>

#include <armnn/utility/Assert.hpp>

namespace armnn
{

void RefLogSoftmaxWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefLogSoftmaxWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefLogSoftmaxWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefLogSoftmaxWorkload_Execute");

    const TensorInfo& inputInfo  = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    std::unique_ptr<Decoder<float>> decoder = MakeDecoder<float>(inputInfo, inputs[0]->Map());
    std::unique_ptr<Encoder<float>> encoder = MakeEncoder<float>(outputInfo, outputs[0]->Map());

    ARMNN_ASSERT(decoder != nullptr);
    ARMNN_ASSERT(encoder != nullptr);

    LogSoftmax(*decoder, *encoder, inputInfo, m_Data.m_Parameters);
}

} // namespace armnn
