//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefBatchNormalizationUint8Workload.hpp"

#include "BatchNormImpl.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

#include <vector>

namespace armnn
{
RefBatchNormalizationUint8Workload::RefBatchNormalizationUint8Workload(
    const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info)
       : Uint8Workload<BatchNormalizationQueueDescriptor>(descriptor, info),
         m_Mean(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Mean))),
         m_Variance(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Variance))),
         m_Beta(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Beta))),
         m_Gamma(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Gamma))) {}

void RefBatchNormalizationUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefBatchNormalizationUint8Workload_Execute");

    const TensorInfo& inputInfo0 = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& varInfo = GetTensorInfo(m_Variance.get());
    const TensorInfo& meanInfo = GetTensorInfo(m_Mean.get());
    const TensorInfo& gammaInfo = GetTensorInfo(m_Gamma.get());
    const TensorInfo& betaInfo = GetTensorInfo(m_Beta.get());
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    auto input = Dequantize(GetInputTensorDataU8(0, m_Data), inputInfo0);
    auto var = Dequantize(m_Variance->GetConstTensor<uint8_t>(), varInfo);
    auto mean = Dequantize(m_Mean->GetConstTensor<uint8_t>(), meanInfo);
    auto gamma = Dequantize(m_Gamma->GetConstTensor<uint8_t>(), gammaInfo);
    auto beta = Dequantize(m_Beta->GetConstTensor<uint8_t>(), betaInfo);

    std::vector<float> results(outputInfo.GetNumElements());
    BatchNormImpl(m_Data, var.data(), mean.data(), gamma.data(), beta.data(), results.data(), input.data());
    Quantize(GetOutputTensorDataU8(0, m_Data), results.data(), outputInfo);
}

} //namespace armnn
