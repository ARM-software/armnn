//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefStackWorkload.hpp"

#include "RefWorkloadUtils.hpp"
#include "Stack.hpp"

#include <Profiling.hpp>

namespace armnn
{

RefStackWorkload::RefStackWorkload(const StackQueueDescriptor& descriptor,
                                   const WorkloadInfo& info)
    : BaseWorkload(descriptor, info)
{}

void RefStackWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefStackWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefStackWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefStackWorkload_Execute");

    // Can perform a simple concatenation when axis == 0
    if (!m_Data.m_Parameters.m_Axis)
    {
        float* output = GetOutputTensorData<float>(0, m_Data);
        ARMNN_ASSERT(output != nullptr);

        unsigned int numInputs = m_Data.m_Parameters.m_NumInputs;
        unsigned int inputLength = GetTensorInfo(inputs[0]).GetNumElements();

        for (unsigned int inputIdx=0; inputIdx<numInputs; ++inputIdx)
        {
            const float* input = GetInputTensorData<float>(inputIdx, m_Data);
            for (unsigned int elmt=0; elmt<inputLength; ++elmt)
            {
                output[(inputIdx * inputLength) + elmt] = input[elmt];
            }
        }
        return;
    }

    std::vector<std::unique_ptr<Decoder<float>>> inputDecoders;
    for (unsigned int i=0; i<inputs.size(); ++i)
    {
        inputDecoders.push_back(MakeDecoder<float>(GetTensorInfo(inputs[i]),
                                                   inputs[i]->Map()));
    }
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]),
                                                                       outputs[0]->Map());

    Stack(m_Data, inputDecoders, *outputEncoder);
}

} // namespace armnn
