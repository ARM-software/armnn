//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "RefConvertFp16ToFp32Workload.hpp"
#include "Half.hpp"
#include "RefWorkloadUtils.hpp"
#include "FloatingPointConverter.hpp"

namespace armnn
{

void RefConvertFp16ToFp32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefConvertFp16ToFp32Workload_Execute");

    const Half* const input = GetInputTensorDataHalf(0, m_Data);
    float* const output = GetOutputTensorDataFloat(0, m_Data);

    unsigned int numElements = GetTensorInfo(m_Data.m_Inputs[0]).GetNumElements();
    armnnUtils::FloatingPointConverter::ConvertFloat16To32(input, numElements, output);
}

} //namespace armnn
