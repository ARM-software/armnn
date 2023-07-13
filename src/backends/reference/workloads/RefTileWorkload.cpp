//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefTileWorkload.hpp"
#include "RefWorkloadUtils.hpp"
#include "Tile.hpp"
#include "Profiling.hpp"

namespace armnn
{

    RefTileWorkload::RefTileWorkload(const TileQueueDescriptor& descriptor, const WorkloadInfo& info)
        : RefBaseWorkload(descriptor, info)
    {}

    void RefTileWorkload::Execute() const
    {
        Execute(m_Data.m_Inputs, m_Data.m_Outputs);
    }

    void RefTileWorkload::ExecuteAsync(ExecutionData& executionData)
    {
        WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
        Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
    }

    void RefTileWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefTileWorkload_Execute");

        const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);

        std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(GetTensorInfo(inputs[0]),
                                                                          inputs[0]->Map());

        std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]),
                                                                           outputs[0]->Map());

        Tile(m_Data.m_Parameters,
             inputInfo,
             *inputDecoder,
             *outputEncoder);
    }

} // namespace armnn