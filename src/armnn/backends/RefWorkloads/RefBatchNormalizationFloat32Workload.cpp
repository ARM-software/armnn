//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "RefBatchNormalizationFloat32Workload.hpp"

#include "BatchNormImpl.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefBatchNormalizationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefBatchNormalizationFloat32Workload_Execute");

    const float* var   = m_Data.m_Variance->GetConstTensor<float>();
    const float* mean  = m_Data.m_Mean->GetConstTensor<float>();
    const float* gamma = m_Data.m_Gamma->GetConstTensor<float>();
    const float* beta  = m_Data.m_Beta->GetConstTensor<float>();

    auto inputData = GetInputTensorDataFloat(0, m_Data);
    auto outputData = GetOutputTensorDataFloat(0, m_Data);

    BatchNormImpl(m_Data, var, mean, gamma, beta, outputData, inputData);
}

} //namespace armnn
