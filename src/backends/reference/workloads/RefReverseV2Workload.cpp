//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefReverseV2Workload.hpp"

#include "ReverseV2Impl.hpp"
#include "RefWorkloadUtils.hpp"
#include "Profiling.hpp"

namespace armnn
{

    RefReverseV2Workload::RefReverseV2Workload(const ReverseV2QueueDescriptor& descriptor, const WorkloadInfo& info)
        : RefBaseWorkload(descriptor, info)
    {}

    void RefReverseV2Workload::Execute() const
    {
        Execute(m_Data.m_Inputs, m_Data.m_Outputs);
    }

    void RefReverseV2Workload::ExecuteAsync(ExecutionData& executionData)
    {
        WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
        Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
    }

    void RefReverseV2Workload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefReverseV2Workload_Execute");

        const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);

        std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(GetTensorInfo(inputs[0]),
                                                                          inputs[0]->Map());

        std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]),
                                                                           outputs[0]->Map());

        ReverseV2(m_Data.m_Parameters,
                  inputInfo,
                  *inputDecoder,
                  *outputEncoder);
    }

} // namespace armnn