//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefBatchNormalizationFloat32Workload.hpp"

#include "BatchNormImpl.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{
RefBatchNormalizationFloat32Workload::RefBatchNormalizationFloat32Workload(
   const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info)
      : Float32Workload<BatchNormalizationQueueDescriptor>(descriptor, info),
        m_Mean(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Mean))),
        m_Variance(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Variance))),
        m_Beta(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Beta))),
        m_Gamma(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Gamma))) {}

void RefBatchNormalizationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefBatchNormalizationFloat32Workload_Execute");

    const float* var   = m_Variance->GetConstTensor<float>();
    const float* mean  = m_Mean->GetConstTensor<float>();
    const float* gamma = m_Gamma->GetConstTensor<float>();
    const float* beta  = m_Beta->GetConstTensor<float>();

    auto inputData = GetInputTensorDataFloat(0, m_Data);
    auto outputData = GetOutputTensorDataFloat(0, m_Data);

    BatchNormImpl(m_Data, var, mean, gamma, beta, outputData, inputData);
}

} //namespace armnn
