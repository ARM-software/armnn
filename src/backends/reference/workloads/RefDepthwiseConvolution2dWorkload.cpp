//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefDepthwiseConvolution2dWorkload.hpp"

#include "ConvImpl.hpp"
#include "RefWorkloadUtils.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"
#include "Profiling.hpp"
#include <ResolveType.hpp>

namespace armnn
{

RefDepthwiseConvolution2dWorkload::RefDepthwiseConvolution2dWorkload(
        const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info)
        : RefBaseWorkload<DepthwiseConvolution2dQueueDescriptor>(descriptor, info)
{
    WorkloadInfo detailsInfo;
    detailsInfo.m_InputTensorInfos = info.m_InputTensorInfos;
    detailsInfo.m_OutputTensorInfos = info.m_OutputTensorInfos;
    detailsInfo.m_WeightsTensorInfo = armnn::Optional<armnn::TensorInfo>(info.m_InputTensorInfos[1]);

    if (descriptor.m_Parameters.m_BiasEnabled)
    {
        detailsInfo.m_BiasTensorInfo = armnn::Optional<armnn::TensorInfo>(info.m_InputTensorInfos[2]);
    }

    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("RefDepthwiseConvolution2dWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         this->GetGuid());
}

void RefDepthwiseConvolution2dWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefDepthwiseConvolution2dWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefDepthwiseConvolution2dWorkload::Execute(std::vector<ITensorHandle*> inputs,
                                                std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDepthwiseConvolution2dWorkload_Execute");

    const TensorShape& inputShape  = GetTensorInfo(inputs[0]).GetShape();
    const TensorShape& outputShape = GetTensorInfo(outputs[0]).GetShape();
    const TensorShape& filterShape = GetTensorInfo(inputs[1]).GetShape();

    std::unique_ptr<Decoder<float>> inputDecoder  = MakeDecoder<float>(GetTensorInfo(inputs[0]), inputs[0]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]), outputs[0]->Map());
    std::unique_ptr<Decoder<float>> filterDecoder = MakeDecoder<float>(GetTensorInfo(inputs[1]), inputs[1]->Map());
    std::unique_ptr<Decoder<float>> biasDecoder{};
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
       biasDecoder = MakeDecoder<float>(GetTensorInfo(inputs[2]), inputs[2]->Map());
    }

    Convolve(inputShape, *inputDecoder, outputShape, *outputEncoder,
             filterShape, *filterDecoder, m_Data.m_Parameters.m_BiasEnabled, biasDecoder.get(),
             m_Data.m_Parameters.m_DataLayout, m_Data.m_Parameters.m_PadTop, m_Data.m_Parameters.m_PadLeft,
             m_Data.m_Parameters.m_StrideX, m_Data.m_Parameters.m_StrideY,
             m_Data.m_Parameters.m_DilationX,
             m_Data.m_Parameters.m_DilationY, true);
}

} //namespace armnn
