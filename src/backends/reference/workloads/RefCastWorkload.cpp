//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefCastWorkload.hpp"
#include "RefWorkloadUtils.hpp"
#include <armnnUtils/FloatingPointConverter.hpp>
#include <ResolveType.hpp>
#include "Encoders.hpp"
#include "Decoders.hpp"

namespace
{
    void Cast(armnn::Decoder<float>& in, armnn::Encoder<float>& out, const uint32_t numElements )
    {
        for (unsigned int i = 0; i < numElements; i++)
        {
            out.Set(in.Get());
            ++in;
            ++out;
        }
    }
}

namespace armnn
{

void RefCastWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefCastWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefCastWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefCastWorkload_Execute");

    TensorInfo inputTensorInfo(GetTensorInfo(inputs[0]));
    TensorInfo outputTensorInfo(GetTensorInfo(outputs[0]));

    // Quantization info should set to default values.
    if (inputTensorInfo.IsQuantized())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }
    if (outputTensorInfo.IsQuantized())
    {
        outputTensorInfo.SetQuantizationScale(1.0f);
        outputTensorInfo.SetQuantizationOffset(0);
    }

    Cast(*MakeDecoder<float>(inputTensorInfo, inputs[0]->Map()),
         *MakeEncoder<float>(outputTensorInfo, outputs[0]->Map()),
         inputTensorInfo.GetNumElements());
}

} //namespace armnn