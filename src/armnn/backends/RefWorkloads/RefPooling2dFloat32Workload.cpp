//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefPooling2dFloat32Workload.hpp"

#include "Pooling2d.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefPooling2dFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefPooling2dFloat32Workload_Execute");

    const TensorInfo& inputInfo0 = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo0 = GetTensorInfo(m_Data.m_Outputs[0]);

    float*       outputData = GetOutputTensorDataFloat(0, m_Data);
    const float* inputData  = GetInputTensorDataFloat(0, m_Data);

    Pooling2d(inputData,
              outputData,
              inputInfo0,
              outputInfo0,
              m_Data.m_Parameters);
}

} //namespace armnn
