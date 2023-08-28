//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefBroadcastToWorkload.hpp"
#include "RefWorkloadUtils.hpp"
#include "Profiling.hpp"
#include "Broadcast.hpp"

#include "Decoders.hpp"
#include "Encoders.hpp"

namespace armnn
{

RefBroadcastToWorkload::RefBroadcastToWorkload(const BroadcastToQueueDescriptor& descriptor, const WorkloadInfo& info)
        : RefBaseWorkload(descriptor, info)
{}

void RefBroadcastToWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefBroadcastToWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefBroadcastToWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT_REF_NAME_GUID("RefBroadcastToWorkload_Execute");
    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    std::unique_ptr<Decoder<float>> input = MakeDecoder<float>(inputInfo, inputs[0]->Map());
    std::unique_ptr<Encoder<float>> output= MakeEncoder<float>(outputInfo, outputs[0]->Map());

    auto broadcastTo = [](float x)
    {
        return x;
    };
    BroadcastLoop(inputInfo.GetShape(), outputInfo.GetShape()).Unroll(broadcastTo,
                                                                      0, *input, *output);
}
} // namespace armnn
