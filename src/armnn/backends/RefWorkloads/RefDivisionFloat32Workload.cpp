//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefDivisionFloat32Workload.hpp"

#include "ArithmeticFunction.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefDivisionFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDivisionFloat32Workload_Execute");

    const TensorShape& inShape0 = GetTensorInfo(m_Data.m_Inputs[0]).GetShape();
    const TensorShape& inShape1 = GetTensorInfo(m_Data.m_Inputs[1]).GetShape();
    const TensorShape& outShape = GetTensorInfo(m_Data.m_Outputs[0]).GetShape();

    float* outputData = GetOutputTensorDataFloat(0, m_Data);
    const float* inputData0 = GetInputTensorDataFloat(0, m_Data);
    const float* inputData1 = GetInputTensorDataFloat(1, m_Data);

    ArithmeticFunction<std::divides<float>>(inShape0, inShape1, outShape, inputData0, inputData1, outputData);
}

} //namespace armnn
