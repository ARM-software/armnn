//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefMeanFloat32Workload.hpp"

#include "Mean.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"
#include "vector"

namespace armnn
{

RefMeanFloat32Workload::RefMeanFloat32Workload(const MeanQueueDescriptor& descriptor, const WorkloadInfo& info)
  :Float32Workload<MeanQueueDescriptor>(descriptor, info) {}


void RefMeanFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefMeanFloat32Workload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);
    const float* inputData = GetInputTensorDataFloat(0, m_Data);
    float* outputData = GetOutputTensorDataFloat(0, m_Data);

    Mean(inputInfo, outputInfo, m_Data.m_Parameters.m_Axis, inputData, outputData);
}

} //namespace armnn


