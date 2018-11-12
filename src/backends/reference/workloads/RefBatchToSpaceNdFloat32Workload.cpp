//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchToSpaceNd.hpp"
#include "Profiling.hpp"
#include "RefBatchToSpaceNdFloat32Workload.hpp"
#include "RefWorkloadUtils.hpp"

namespace armnn
{

void RefBatchToSpaceNdFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefBatchToSpaceNdFloat32Workload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);
    const float* inputData = GetInputTensorDataFloat(0, m_Data);
    float* outputData = GetOutputTensorDataFloat(0, m_Data);

    BatchToSpaceNd(m_Data.m_Parameters.m_DataLayout, inputInfo, outputInfo, m_Data.m_Parameters.m_BlockShape,
                   m_Data.m_Parameters.m_Crops, inputData, outputData);
}


} //namespace armnn