//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
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
    RefBaseWorkload<TransposeConvolution2dQueueDescriptor>(descriptor, info)
{
    // set up weights decoder
    m_Weights = std::make_unique<ScopedTensorHandle>(*(descriptor.m_Weight));
    const TensorInfo& weightsInfo = m_Weights->GetTensorInfo();

    m_WeightsDecoder = MakeDecoder<float>(weightsInfo, m_Weights->Map(true));
    m_WeightsShape   = weightsInfo.GetShape();

    // set up biases decoder
    if (descriptor.m_Parameters.m_BiasEnabled)
    {
        m_Biases = std::make_unique<ScopedTensorHandle>(*(descriptor.m_Bias));
        const TensorInfo& biasesInfo = m_Biases->GetTensorInfo();
        m_BiasesDecoder = MakeDecoder<float>(biasesInfo, m_Biases->Map(true));
    }
}

void RefTransposeConvolution2dWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefTransposeConvolution2dWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefTransposeConvolution2dWorkload::Execute(std::vector<ITensorHandle*> inputs,
                                                std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefTransposeConvolution2dWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(inputInfo, inputs[0]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(outputInfo, outputs[0]->Map());

    TransposeConvolution2dImpl(m_Data.m_Parameters,
                               inputInfo.GetShape(),
                               *inputDecoder,
                               outputInfo.GetShape(),
                               *outputEncoder,
                               m_WeightsShape,
                               *m_WeightsDecoder,
                               m_BiasesDecoder.get());
}

} // namespace armnn