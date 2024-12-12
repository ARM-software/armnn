//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
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

    void RefReverseV2Workload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
    {
        ARMNN_SCOPED_PROFILING_EVENT_REF_NAME_GUID("RefReverseV2Workload_Execute");

        const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
        const TensorInfo& axisInfo = GetTensorInfo(inputs[1]);

        std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(GetTensorInfo(inputs[0]),
                                                                          inputs[0]->Map());

        std::unique_ptr<Decoder<int>> axisDecoder = MakeDecoder<int>(GetTensorInfo(inputs[1]),
                                                                     inputs[1]->Map());

        std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]),
                                                                           outputs[0]->Map());

        ReverseV2(inputInfo,
                  axisInfo,
                  *inputDecoder,
                  *axisDecoder,
                  *outputEncoder);
    }

} // namespace armnn