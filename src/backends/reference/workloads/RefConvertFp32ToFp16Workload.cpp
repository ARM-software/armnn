//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefConvertFp32ToFp16Workload.hpp"

#include "FloatingPointConverter.hpp"
#include "RefWorkloadUtils.hpp"
#include "Profiling.hpp"

#include "Half.hpp"

namespace armnn
{

void RefConvertFp32ToFp16Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefConvertFp32ToFp16Workload_Execute");

    const float* const input = GetInputTensorDataFloat(0, m_Data);
    Half*  const output = GetOutputTensorDataHalf(0, m_Data);

    // convert Fp32 input to Fp16 output
    unsigned int numElements = GetTensorInfo(m_Data.m_Inputs[0]).GetNumElements();
    armnnUtils::FloatingPointConverter::ConvertFloat32To16(input, numElements, output);
}

} //namespace armnn
