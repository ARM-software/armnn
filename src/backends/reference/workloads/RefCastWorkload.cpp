//
// Copyright © 2021-2024 Arm Ltd and Contributors. All rights reserved.
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
    void Cast(armnn::Decoder<float>& in, armnn::Encoder<float>& out,
              const uint32_t numElements, const armnn::DataType OutputDataType)
    {
        for (unsigned int i = 0; i < numElements; ++i)
        {
            switch (OutputDataType)
            {
                case armnn::DataType::Float32:
                case armnn::DataType::Float16:
                case armnn::DataType::BFloat16:
                    out.Set(in.Get());
                    break;
                default:
                    out.Set(std::floor(in.Get()));
                    break;
            }
            ++in;
            ++out;
        }
    }


    // Cast Float to Int64
    void Cast(armnn::Decoder<float>& in, armnn::Encoder<double_t>& out,
              const uint32_t numElements, const armnn::DataType)
    {
        for (unsigned int i = 0; i < numElements; ++i)
        {
            out.Set(in.Get());
            ++in;
            ++out;
        }
    }

    // Cast Int64 To Float
    void Cast(armnn::Decoder<double_t>& in, armnn::Encoder<float>& out,
              const uint32_t numElements, const armnn::DataType)
    {
        for (unsigned int i = 0; i < numElements; ++i)
        {
            out.Set(static_cast<float>(in.Get()));
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
    ARMNN_SCOPED_PROFILING_EVENT_REF_NAME_GUID("RefCastWorkload_Execute");

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

    if(inputTensorInfo.GetDataType() == DataType::Signed64)
    {
        Cast(*MakeDecoder<double_t>(inputTensorInfo, inputs[0]->Map()),
             *MakeEncoder<float>(outputTensorInfo, outputs[0]->Map()),
             inputTensorInfo.GetNumElements(),
             outputTensorInfo.GetDataType());
    }
    else if(outputTensorInfo.GetDataType() == DataType::Signed64)
    {
        Cast(*MakeDecoder<float>(inputTensorInfo, inputs[0]->Map()),
             *MakeEncoder<double_t>(outputTensorInfo, outputs[0]->Map()),
             inputTensorInfo.GetNumElements(),
             outputTensorInfo.GetDataType());
    }
    else
    {
        Cast(*MakeDecoder<float>(inputTensorInfo, inputs[0]->Map()),
             *MakeEncoder<float>(outputTensorInfo, outputs[0]->Map()),
             inputTensorInfo.GetNumElements(),
             outputTensorInfo.GetDataType());
    }
}

} //namespace armnn