//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "RefMultiplicationFloat32Workload.hpp"

#include "Multiplication.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefMultiplicationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefMultiplicationFloat32Workload_Execute");

    const TensorInfo& inputInfo0 = GetTensorInfo(m_Data.m_Inputs[0]);

    float* outputData = GetOutputTensorDataFloat(0, m_Data);
    const float* inputData0 = GetInputTensorDataFloat(0, m_Data);
    const float* inputData1 = GetInputTensorDataFloat(1, m_Data);
    Multiplication(inputData0, inputData1, inputInfo0.GetNumElements(), outputData);
}

} //namespace armnn
