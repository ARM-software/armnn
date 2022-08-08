//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefBatchMatMulWorkload.hpp"

#include "BatchMatMulImpl.hpp"
#include "RefWorkloadUtils.hpp"
#include "Profiling.hpp"

namespace armnn
{

RefBatchMatMulWorkload::RefBatchMatMulWorkload(const BatchMatMulQueueDescriptor& descriptor, const WorkloadInfo& info)
    : RefBaseWorkload(descriptor, info)
{}

void RefBatchMatMulWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefBatchMatMulWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefBatchMatMulWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefBatchMatMulWorkload_Execute");

    const TensorInfo& inputXInfo = GetTensorInfo(inputs[0]);
    const TensorInfo& inputYInfo = GetTensorInfo(inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    std::unique_ptr<Decoder<float>> inputXDecoder = MakeDecoder<float>(GetTensorInfo(inputs[0]),
                                                                       inputs[0]->Map());

    std::unique_ptr<Decoder<float>> inputYDecoder = MakeDecoder<float>(GetTensorInfo(inputs[1]),
                                                                       inputs[1]->Map());

    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]),
                                                                       outputs[0]->Map());

    auto bmm = BatchMatMul(m_Data.m_Parameters,
                           inputXInfo,
                           inputYInfo,
                           outputInfo,
                           *inputXDecoder,
                           *inputYDecoder,
                           *outputEncoder);
}

} // namespace armnn