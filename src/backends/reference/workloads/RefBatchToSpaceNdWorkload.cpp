//
// Copyright © 2017,2019,2021-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefBatchToSpaceNdWorkload.hpp"
#include "BatchToSpaceNd.hpp"

#include "RefWorkloadUtils.hpp"

namespace armnn
{

void RefBatchToSpaceNdWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefBatchToSpaceNdWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT_REF_NAME_GUID("RefBatchToSpaceNdWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    std::unique_ptr<Decoder<float>> inputDecoder  = MakeDecoder<float>(inputInfo, inputs[0]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(outputInfo, outputs[0]->Map());

    BatchToSpaceNd(inputInfo, outputInfo, m_Data.m_Parameters, *inputDecoder, *outputEncoder);
}


} //namespace armnn
