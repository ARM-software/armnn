//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefFullyConnectedUint8Workload.hpp"

#include "FullyConnected.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

#include <vector>

namespace armnn
{
RefFullyConnectedUint8Workload::RefFullyConnectedUint8Workload(
    const FullyConnectedQueueDescriptor& descriptor, const WorkloadInfo& info)
     : Uint8Workload<FullyConnectedQueueDescriptor>(descriptor, info),
        m_Weight(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Weight))),
        m_Bias(descriptor.m_Parameters.m_BiasEnabled
               ? std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Bias)) : nullptr) {}

void RefFullyConnectedUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefFullyConnectedUint8Workload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    const uint8_t* weightData = m_Weight->GetConstTensor<uint8_t>();

    auto dequant = Dequantize(GetInputTensorDataU8(0, m_Data), inputInfo);

    auto weight = Dequantize(weightData, m_Weight->GetTensorInfo());

    std::vector<float> results(outputInfo.GetNumElements());

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        const int32_t* biasData = m_Bias->GetConstTensor<int32_t>();
        auto           bias     = Dequantize(biasData, m_Bias->GetTensorInfo());

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
