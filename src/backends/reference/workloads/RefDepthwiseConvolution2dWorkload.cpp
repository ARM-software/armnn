//
// Copyright © 2017 Arm Ltd. All rights reserved.
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

void RefDepthwiseConvolution2dWorkload::PostAllocationConfigure()
{
    PostAllocationConfigure(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefDepthwiseConvolution2dWorkload::PostAllocationConfigure(std::vector<ITensorHandle*> inputs,
                                                                std::vector<ITensorHandle*> outputs)
{
    IgnoreUnused(outputs);

    const TensorInfo& rFilterInfo = GetTensorInfo(inputs[1]);
    m_FilterShape = rFilterInfo.GetShape();
    m_FilterDecoder = MakeDecoder<float>(rFilterInfo);

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        const TensorInfo& biasInfo = GetTensorInfo(inputs[2]);
        m_BiasDecoder = MakeDecoder<float>(biasInfo);
    }
}

void RefDepthwiseConvolution2dWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefDepthwiseConvolution2dWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    PostAllocationConfigure(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);

    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefDepthwiseConvolution2dWorkload::Execute(std::vector<ITensorHandle*> inputs,
                                                std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDepthwiseConvolution2dWorkload_Execute");
    std::unique_ptr<Decoder<float>> pBiasDecoder{};

    std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(GetTensorInfo(inputs[0]), inputs[0]->Map());
    std::unique_ptr<Encoder<float>> OutputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]), outputs[0]->Map());

    const TensorShape& inputShape = GetTensorInfo(inputs[0]).GetShape();
    const TensorShape& outputShape = GetTensorInfo(outputs[0]).GetShape();

    m_FilterDecoder->Reset(inputs[1]->Map());
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasDecoder->Reset(inputs[2]->Map());
    }

    Convolve(inputShape, *inputDecoder, outputShape, *OutputEncoder,
             m_FilterShape, *m_FilterDecoder, m_Data.m_Parameters.m_BiasEnabled, m_BiasDecoder.get(),
             m_Data.m_Parameters.m_DataLayout, m_Data.m_Parameters.m_PadTop, m_Data.m_Parameters.m_PadLeft,
             m_Data.m_Parameters.m_StrideX, m_Data.m_Parameters.m_StrideY,
             m_Data.m_Parameters.m_DilationX,
             m_Data.m_Parameters.m_DilationY, true);
}

} //namespace armnn
