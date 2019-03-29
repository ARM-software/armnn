//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefDequantizeWorkload.hpp"
#include "RefWorkloadUtils.hpp"

namespace armnn
{

void RefDequantizeWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDequantizeWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const DataType& inputDataType = inputInfo.GetDataType();

    float* outputData = GetOutputTensorData<float>(0, m_Data);

    switch (inputDataType)
    {
        case DataType::QuantisedAsymm8:
            Dequantize<uint8_t>(GetInputTensorData<uint8_t>(0, m_Data), outputData, inputInfo);
            break;
        case DataType::QuantisedSymm16:
            Dequantize<int16_t>(GetInputTensorData<int16_t>(0, m_Data), outputData, inputInfo);
            break;
        default:
            throw InvalidArgumentException("RefDequantizeWorkload: Unsupported input data type");
    }
}

} // namespace armnn
