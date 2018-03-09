//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "RefBatchNormalizationUint8Workload.hpp"

#include "BatchNormImpl.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

#include <vector>

namespace armnn
{

void RefBatchNormalizationUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefBatchNormalizationUint8Workload_Execute");

    const TensorInfo& inputInfo0 = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& varInfo = GetTensorInfo(m_Data.m_Variance);
    const TensorInfo& meanInfo = GetTensorInfo(m_Data.m_Mean);
    const TensorInfo& gammaInfo = GetTensorInfo(m_Data.m_Gamma);
    const TensorInfo& betaInfo = GetTensorInfo(m_Data.m_Beta);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    auto input = Dequantize(GetInputTensorDataU8(0, m_Data), inputInfo0);
    auto var = Dequantize(m_Data.m_Variance->GetConstTensor<uint8_t>(), varInfo);
    auto mean = Dequantize(m_Data.m_Mean->GetConstTensor<uint8_t>(), meanInfo);
    auto gamma = Dequantize(m_Data.m_Gamma->GetConstTensor<uint8_t>(), gammaInfo);
    auto beta = Dequantize(m_Data.m_Beta->GetConstTensor<uint8_t>(), betaInfo);

    std::vector<float> results(outputInfo.GetNumElements());
    BatchNormImpl(m_Data, var.data(), mean.data(), gamma.data(), beta.data(), results.data(), input.data());
    Quantize(GetOutputTensorDataU8(0, m_Data), results.data(), outputInfo);
}

} //namespace armnn
