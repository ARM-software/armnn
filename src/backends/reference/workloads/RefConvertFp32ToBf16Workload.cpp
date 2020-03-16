//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefConvertFp32ToBf16Workload.hpp"
#include "RefWorkloadUtils.hpp"

#include <armnnUtils/FloatingPointConverter.hpp>

#include <BFloat16.hpp>

namespace armnn
{

void RefConvertFp32ToBf16Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefConvertFp32ToBf16Workload_Execute");

    const float* const input = GetInputTensorDataFloat(0, m_Data);
    BFloat16* const output = GetOutputTensorDataBFloat16(0, m_Data);

    unsigned int numElements = GetTensorInfo(m_Data.m_Inputs[0]).GetNumElements();
    armnnUtils::FloatingPointConverter::ConvertFloat32ToBFloat16(input, numElements, output);
}

} //namespace armnn
