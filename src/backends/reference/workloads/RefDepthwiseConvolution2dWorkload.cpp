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
    m_Weight = std::make_unique<ScopedTensorHandle>(*(descriptor.m_Weight));
    const TensorInfo& rFilterInfo = m_Weight->GetTensorInfo();
    m_FilterShape = rFilterInfo.GetShape();
    m_FilterDecoder = MakeDecoder<float>(rFilterInfo, m_Weight->Map(true));

    if (descriptor.m_Parameters.m_BiasEnabled)
    {
        m_Bias = std::make_unique<ScopedTensorHandle>(*(descriptor.m_Bias));
        const TensorInfo& biasInfo = m_Bias->GetTensorInfo();
        m_BiasDecoder = MakeDecoder<float>(biasInfo, m_Bias->Map(true));
    }
}

void RefDepthwiseConvolution2dWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefDepthwiseConvolution2dWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
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

    Convolve(inputShape, *inputDecoder, outputShape, *OutputEncoder,
             m_FilterShape, *m_FilterDecoder, m_Data.m_Parameters.m_BiasEnabled, m_BiasDecoder.get(),
             m_Data.m_Parameters.m_DataLayout, m_Data.m_Parameters.m_PadTop, m_Data.m_Parameters.m_PadLeft,
             m_Data.m_Parameters.m_StrideX, m_Data.m_Parameters.m_StrideY,
             m_Data.m_Parameters.m_DilationX,
             m_Data.m_Parameters.m_DilationY, true);
}

} //namespace armnn
