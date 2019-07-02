//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefTransposeConvolution2dWorkload.hpp"

#include "RefWorkloadUtils.hpp"
#include "TransposeConvolution2d.hpp"

#include <Profiling.hpp>

namespace armnn
{

RefTransposeConvolution2dWorkload::RefTransposeConvolution2dWorkload(
    const TransposeConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) :
    BaseWorkload<TransposeConvolution2dQueueDescriptor>(descriptor, info)
{
    // set up weights decoder
    m_Weights = std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Weight));
    const TensorInfo& weightsInfo = m_Weights->GetTensorInfo();

    m_WeightsDecoder = MakeDecoder<float>(weightsInfo, m_Weights->Map(true));
    m_WeightsShape   = weightsInfo.GetShape();

    // set up biases decoder
    if (descriptor.m_Parameters.m_BiasEnabled)
    {
        m_Biases = std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Bias));
        const TensorInfo& biasesInfo = m_Biases->GetTensorInfo();
        m_BiasesDecoder = MakeDecoder<float>(biasesInfo, m_Biases->Map(true));
    }
}

void RefTransposeConvolution2dWorkload::PostAllocationConfigure()
{
    // set up input decoder
    const ITensorHandle* input  = m_Data.m_Inputs[0];
    const TensorInfo& inputInfo = GetTensorInfo(input);

    m_InputShape   = inputInfo.GetShape();
    m_InputDecoder = MakeDecoder<float>(inputInfo);

    // set up output encoder
    ITensorHandle* output        = m_Data.m_Outputs[0];
    const TensorInfo& outputInfo = GetTensorInfo(output);

    m_OutputShape   = outputInfo.GetShape();
    m_OutputEncoder = MakeEncoder<float>(outputInfo);
}

void RefTransposeConvolution2dWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefTransposeConvolution2dWorkload_Execute");

    m_InputDecoder->Reset(m_Data.m_Inputs[0]->Map());
    m_OutputEncoder->Reset(m_Data.m_Outputs[0]->Map());

    TransposeConvolution2dImpl(m_Data.m_Parameters,
                               m_InputShape,
                               *m_InputDecoder,
                               m_OutputShape,
                               *m_OutputEncoder,
                               m_WeightsShape,
                               *m_WeightsDecoder,
                               m_BiasesDecoder.get());
}

} // namespace armnn