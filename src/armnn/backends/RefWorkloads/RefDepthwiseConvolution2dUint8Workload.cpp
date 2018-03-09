//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "RefDepthwiseConvolution2dUint8Workload.hpp"

#include "ConvImpl.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefDepthwiseConvolution2dUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDepthwiseConvolution2dUint8Workload_Execute");

    const uint8_t* inputData = GetInputTensorDataU8(0, m_Data);
    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const uint8_t* weightsData = m_Data.m_Weight->template GetConstTensor<uint8_t>();
    const TensorInfo& weightsInfo = GetTensorInfo(m_Data.m_Weight);
    const int32_t* biasData = m_Data.m_Parameters.m_BiasEnabled ?
        m_Data.m_Bias->template GetConstTensor<int32_t>() :
        nullptr;
    uint8_t* outputData = GetOutputTensorDataU8(0, m_Data);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    ConvImpl<armnn::DepthwiseConvolution2dQueueDescriptor, uint8_t, int32_t, int32_t>(
        m_Data,
        inputData, inputInfo.GetQuantizationScale(),  inputInfo.GetQuantizationOffset(),
        weightsData, weightsInfo.GetQuantizationScale(), weightsInfo.GetQuantizationOffset(),
        biasData,
        outputData, outputInfo.GetQuantizationScale(), outputInfo.GetQuantizationOffset(), true);
}

} //namespace armnn
