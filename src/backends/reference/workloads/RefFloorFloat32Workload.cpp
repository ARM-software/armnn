//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefFloorFloat32Workload.hpp"

#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefFloorFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefFloorFloat32Workload_Execute");

    const float* const input = GetInputTensorDataFloat(0, m_Data);
    float* const output = GetOutputTensorDataFloat(0, m_Data);

    unsigned int numElements = GetTensorInfo(m_Data.m_Inputs[0]).GetNumElements();
    for (unsigned int i = 0; i < numElements; ++i)
    {
        output[i] = floorf(input[i]);
    }
}

} //namespace armnn
