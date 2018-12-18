//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefDepthwiseConvolution2dUint8Workload.hpp"

#include "ConvImpl.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

RefDepthwiseConvolution2dUint8Workload::RefDepthwiseConvolution2dUint8Workload(
        const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info)
        : Uint8Workload<DepthwiseConvolution2dQueueDescriptor>(descriptor, info),
          m_Weight(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Weight))),
          m_Bias(descriptor.m_Parameters.m_BiasEnabled
                 ? std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Bias)) : nullptr) {}

void RefDepthwiseConvolution2dUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDepthwiseConvolution2dUint8Workload_Execute");

    const uint8_t* inputData = GetInputTensorDataU8(0, m_Data);
    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const uint8_t* weightsData = m_Weight->template GetConstTensor<uint8_t>();
    const TensorInfo& weightsInfo = GetTensorInfo(m_Weight.get());
    const int32_t* biasData = m_Data.m_Parameters.m_BiasEnabled ? m_Bias->template GetConstTensor<int32_t>() : nullptr;
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);
    const TensorInfo& filterInfo = m_Weight->GetTensorInfo();

    ConvImpl<armnn::DepthwiseConvolution2dQueueDescriptor, uint8_t, int32_t, int32_t>(
        m_Data,
        inputData, inputInfo.GetQuantizationScale(),  inputInfo.GetQuantizationOffset(),
        weightsData, weightsInfo.GetQuantizationScale(), weightsInfo.GetQuantizationOffset(),
        biasData,
        outputInfo.GetQuantizationScale(), outputInfo.GetQuantizationOffset(), filterInfo, true);
}

} //namespace armnn
