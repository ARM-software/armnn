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
        : BaseWorkload<DepthwiseConvolution2dQueueDescriptor>(descriptor, info)
{
    m_Weight = std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Weight));
    const TensorInfo& rFilterInfo = m_Weight->GetTensorInfo();
    m_FilterShape = rFilterInfo.GetShape();
    m_FilterDecoder = MakeDecoder<float>(rFilterInfo, m_Weight->Map(true));

    if (descriptor.m_Parameters.m_BiasEnabled)
    {
        m_Bias = std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Bias));
        const TensorInfo& biasInfo = m_Bias->GetTensorInfo();
        m_BiasDecoder = MakeDecoder<float>(biasInfo, m_Bias->Map(true));
    }
}

void RefDepthwiseConvolution2dWorkload::PostAllocationConfigure()
{
    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    m_InputShape = inputInfo.GetShape();
    m_InputDecoder = MakeDecoder<float>(inputInfo);

    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);
    m_OutputShape = outputInfo.GetShape();
    m_OutputEncoder = MakeEncoder<float>(outputInfo);
}

void RefDepthwiseConvolution2dWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDepthwiseConvolution2dWorkload_Execute");
    std::unique_ptr<Decoder<float>> pBiasDecoder{};

    m_InputDecoder->Reset(m_Data.m_Inputs[0]->Map());
    m_OutputEncoder->Reset(m_Data.m_Outputs[0]->Map());

    Convolve(m_InputShape, *m_InputDecoder, m_OutputShape, *m_OutputEncoder,
             m_FilterShape, *m_FilterDecoder, m_Data.m_Parameters.m_BiasEnabled, m_BiasDecoder.get(),
             m_Data.m_Parameters.m_DataLayout, m_Data.m_Parameters.m_PadTop, m_Data.m_Parameters.m_PadLeft,
             m_Data.m_Parameters.m_StrideX, m_Data.m_Parameters.m_StrideY,
             m_Data.m_Parameters.m_DilationX,
             m_Data.m_Parameters.m_DilationY, true);
}

} //namespace armnn
