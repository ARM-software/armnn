//
// Copyright © 2017 Arm Ltd. All rights reserved.
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
    : BaseWorkload(descriptor, info)
    , m_Mean    (std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Mean)))
    , m_Variance(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Variance)))
    , m_Beta    (std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Beta)))
    , m_Gamma   (std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Gamma)))
{}

void RefBatchNormalizationWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefBatchNormalizationWorkload_Execute");

    std::unique_ptr<Decoder<float>> meanDecoder = MakeDecoder<float>(GetTensorInfo(m_Mean.get()),
                                                                     m_Mean.get()->Map(true));
    std::unique_ptr<Decoder<float>> varianceDecoder = MakeDecoder<float>(GetTensorInfo(m_Variance.get()),
                                                                         m_Variance.get()->Map(true));
    std::unique_ptr<Decoder<float>> gammaDecoder = MakeDecoder<float>(GetTensorInfo(m_Gamma.get()),
                                                                      m_Gamma.get()->Map(true));
    std::unique_ptr<Decoder<float>> betaDecoder = MakeDecoder<float>(GetTensorInfo(m_Beta.get()),
                                                                     m_Beta.get()->Map(true));
    std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(GetTensorInfo(m_Data.m_Inputs[0]),
                                                                      m_Data.m_Inputs[0]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(m_Data.m_Outputs[0]),
                                                                       m_Data.m_Outputs[0]->Map());

    BatchNormImpl(m_Data, *meanDecoder, *varianceDecoder, *betaDecoder, *gammaDecoder, *inputDecoder, *outputEncoder);
}

} //namespace armnn
