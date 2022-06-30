//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
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
    : RefBaseWorkload(descriptor, info)
{}

void RefStackWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefStackWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefStackWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefStackWorkload_Execute");

    const TensorInfo& inputInfo  = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    std::vector<std::unique_ptr<Decoder<float>>> inputDecoders;
    for (unsigned int i=0; i<inputs.size(); ++i)
    {
        inputDecoders.push_back(MakeDecoder<float>(GetTensorInfo(inputs[i]),
                                                   inputs[i]->Map()));
    }
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]),
                                                                       outputs[0]->Map());

    Stack(m_Data, inputDecoders, *outputEncoder, inputInfo, outputInfo);
}

} // namespace armnn
