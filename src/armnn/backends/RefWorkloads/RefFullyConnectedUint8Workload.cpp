//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "RefFullyConnectedUint8Workload.hpp"

#include "FullyConnected.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

#include <vector>

namespace armnn
{

void RefFullyConnectedUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefFullyConnectedUint8Workload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    const uint8_t* weightData = m_Data.m_Weight->GetConstTensor<uint8_t>();

    auto dequant = Dequantize(GetInputTensorDataU8(0, m_Data), inputInfo);

    auto weight = Dequantize(weightData, m_Data.m_Weight->GetTensorInfo());

    std::vector<float> results(inputInfo.GetNumElements());

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        const int32_t* biasData = m_Data.m_Bias->GetConstTensor<int32_t>();
        auto           bias     = Dequantize(biasData, m_Data.m_Bias->GetTensorInfo());

        FullyConnected(dequant.data(),
                       results.data(),
                       inputInfo,
                       outputInfo,
                       weight.data(),
                       bias.data(),
                       m_Data.m_Parameters.m_TransposeWeightMatrix);
    }
    else
    {
        FullyConnected(dequant.data(),
                       results.data(),
                       inputInfo,
                       outputInfo,
                       weight.data(),
                       nullptr,
                       m_Data.m_Parameters.m_TransposeWeightMatrix);
    }

    Quantize(GetOutputTensorDataU8(0, m_Data), results.data(), outputInfo);
}

} //namespace armnn
