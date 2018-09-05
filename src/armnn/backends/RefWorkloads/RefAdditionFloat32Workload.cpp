//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefAdditionFloat32Workload.hpp"

#include "Addition.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefAdditionFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefAdditionFloat32Workload_Execute");

    const TensorShape& inShape0 = GetTensorInfo(m_Data.m_Inputs[0]).GetShape();
    const TensorShape& inShape1 = GetTensorInfo(m_Data.m_Inputs[1]).GetShape();
    const TensorShape& outShape = GetTensorInfo(m_Data.m_Outputs[0]).GetShape();

    const float* inData0 = GetInputTensorDataFloat(0, m_Data);
    const float* inData1 = GetInputTensorDataFloat(1, m_Data);
    float* outData = GetOutputTensorDataFloat(0, m_Data);

    Addition(inShape0, inShape1, outShape, inData0, inData1, outData);
}

} //namespace armnn
