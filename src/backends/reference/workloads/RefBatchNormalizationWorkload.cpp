//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefBatchNormalizationWorkload.hpp"

#include "BatchNormImpl.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

RefBatchNormalizationWorkload::RefBatchNormalizationWorkload(const BatchNormalizationQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info)
    : RefBaseWorkload(descriptor, info)
    , m_Mean    (std::make_unique<ScopedTensorHandle>(*(descriptor.m_Mean)))
    , m_Variance(std::make_unique<ScopedTensorHandle>(*(descriptor.m_Variance)))
    , m_Beta    (std::make_unique<ScopedTensorHandle>(*(descriptor.m_Beta)))
    , m_Gamma   (std::make_unique<ScopedTensorHandle>(*(descriptor.m_Gamma)))
{}

void RefBatchNormalizationWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefBatchNormalizationWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefBatchNormalizationWorkload::Execute(std::vector<ITensorHandle*> inputs,
                                            std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefBatchNormalizationWorkload_Execute");

    std::unique_ptr<Decoder<float>> meanDecoder     = MakeDecoder<float>(m_Mean->GetTensorInfo(),
                                                                         m_Mean->Map(true));
    std::unique_ptr<Decoder<float>> varianceDecoder = MakeDecoder<float>(m_Variance->GetTensorInfo(),
                                                                         m_Variance->Map(true));
    std::unique_ptr<Decoder<float>> gammaDecoder    = MakeDecoder<float>(m_Gamma->GetTensorInfo(),
                                                                         m_Gamma->Map(true));
    std::unique_ptr<Decoder<float>> betaDecoder     = MakeDecoder<float>(m_Beta->GetTensorInfo(),
                                                                         m_Beta->Map(true));
    std::unique_ptr<Decoder<float>> inputDecoder    = MakeDecoder<float>(GetTensorInfo(inputs[0]),
                                                                         inputs[0]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder   = MakeEncoder<float>(GetTensorInfo(outputs[0]),
                                                                         outputs[0]->Map());

    BatchNormImpl(m_Data, *meanDecoder, *varianceDecoder, *betaDecoder, *gammaDecoder, *inputDecoder, *outputEncoder);
}

} // namespace armnn
