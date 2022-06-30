//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchToSpaceNd.hpp"
#include "Profiling.hpp"
#include "RefBatchToSpaceNdWorkload.hpp"
#include "RefWorkloadUtils.hpp"

namespace armnn
{

void RefBatchToSpaceNdWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefBatchToSpaceNdWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefBatchToSpaceNdWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefBatchToSpaceNdWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    std::unique_ptr<Decoder<float>> inputDecoder  = MakeDecoder<float>(inputInfo, inputs[0]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(outputInfo, outputs[0]->Map());

    BatchToSpaceNd(m_Data.m_Parameters.m_DataLayout, inputInfo, outputInfo, m_Data.m_Parameters.m_BlockShape,
                   m_Data.m_Parameters.m_Crops, *inputDecoder, *outputEncoder);
}


} //namespace armnn
