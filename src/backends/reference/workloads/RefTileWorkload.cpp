//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
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
        auto inputDataType = GetTensorInfo(m_Data.m_Inputs[0]).GetDataType();
        if(inputDataType == DataType::Signed64)
        {
            Execute<double_t>(m_Data.m_Inputs, m_Data.m_Outputs);
        }
        else
        {
            Execute<float>(m_Data.m_Inputs, m_Data.m_Outputs);
        }
    }

    void RefTileWorkload::ExecuteAsync(ExecutionData& executionData)
    {
        auto* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
        auto inputDataType = GetTensorInfo(workingMemDescriptor->m_Inputs[0]).GetDataType();
        if(inputDataType == DataType::Signed64)
        {
            Execute<double_t>(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
        }
        else
        {
            Execute<float>(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
        }
    }

    template <typename T>
    void RefTileWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
    {
        ARMNN_SCOPED_PROFILING_EVENT_REF_NAME_GUID("RefTileWorkload_Execute");

        const TensorInfo& inputInfo  = GetTensorInfo(inputs[0]);
        const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

        std::unique_ptr<Decoder<T>> inputDecoder  = MakeDecoder<T>(inputInfo,
                                                                   inputs[0]->Map());
        std::unique_ptr<Encoder<T>> outputEncoder = MakeEncoder<T>(outputInfo,
                                                                   outputs[0]->Map());

        Tile<T, T>(m_Data.m_Parameters,
                   inputInfo,
                   *inputDecoder,
                   *outputEncoder);
    }

} // namespace armnn