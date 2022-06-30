//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefInstanceNormalizationWorkload.hpp"

#include "InstanceNorm.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

RefInstanceNormalizationWorkload::RefInstanceNormalizationWorkload(
    const InstanceNormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info)
    : RefBaseWorkload<InstanceNormalizationQueueDescriptor>(descriptor, info) {}

void RefInstanceNormalizationWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefInstanceNormalizationWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefInstanceNormalizationWorkload::Execute(std::vector<ITensorHandle*> inputs,
                                               std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefInstanceNormalizationWorkload_Execute");

    std::unique_ptr<Decoder<float>> inputDecoder  = MakeDecoder<float>(GetTensorInfo(inputs[0]),
                                                                       inputs[0]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]),
                                                                       outputs[0]->Map());
    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);

    InstanceNorm(m_Data, inputInfo, *inputDecoder, *outputEncoder);
}

} // namespace armnn
