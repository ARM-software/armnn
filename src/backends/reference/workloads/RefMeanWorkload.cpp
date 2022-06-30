//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefMeanWorkload.hpp"

#include "Reduce.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

#include <vector>

namespace armnn
{

RefMeanWorkload::RefMeanWorkload(const MeanQueueDescriptor& descriptor, const WorkloadInfo& info)
  :RefBaseWorkload<MeanQueueDescriptor>(descriptor, info) {}

void RefMeanWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefMeanWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefMeanWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefMeanWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    auto inputDecoder  = MakeDecoder<float>(inputInfo,  inputs[0]->Map());
    auto outputEncoder = MakeEncoder<float>(outputInfo, outputs[0]->Map());

    Reduce(inputInfo,
           outputInfo,
           *inputDecoder,
           *outputEncoder,
           m_Data.m_Parameters.m_Axis,
           armnn::ReduceOperation::Mean);
}

} //namespace armnn
