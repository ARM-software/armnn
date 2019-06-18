//
// Copyright © 2017 Arm Ltd. All rights reserved.
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
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefBatchToSpaceNdWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    std::unique_ptr<Decoder<float>> inputDecoder  = MakeDecoder<float>(inputInfo, m_Data.m_Inputs[0]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());

    BatchToSpaceNd(m_Data.m_Parameters.m_DataLayout, inputInfo, outputInfo, m_Data.m_Parameters.m_BlockShape,
                   m_Data.m_Parameters.m_Crops, *inputDecoder, *outputEncoder);
}


} //namespace armnn
