//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefConvertBf16ToFp32Workload.hpp"
#include "RefWorkloadUtils.hpp"

#include <armnnUtils/FloatingPointConverter.hpp>

#include <BFloat16.hpp>

namespace armnn
{

void RefConvertBf16ToFp32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefConvertBf16ToFp32Workload_Execute");

    const BFloat16* const input = GetInputTensorDataBFloat16(0, m_Data);
    float* const output = GetOutputTensorDataFloat(0, m_Data);

    unsigned int numElements = GetTensorInfo(m_Data.m_Inputs[0]).GetNumElements();
    armnnUtils::FloatingPointConverter::ConvertBFloat16ToFloat32(input, numElements, output);
}

} //namespace armnn
