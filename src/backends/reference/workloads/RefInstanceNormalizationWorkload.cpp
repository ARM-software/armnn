//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefInstanceNormalizationWorkload.hpp"

#include "InstanceNorm.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

RefInstanceNormalizationWorkload::RefInstanceNormalizationWorkload(
    const InstanceNormalizationQueueDescriptor& descriptor,
    const WorkloadInfo& info)
    : BaseWorkload<InstanceNormalizationQueueDescriptor>(descriptor, info) {}

void RefInstanceNormalizationWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefInstanceNormalizationWorkload_Execute");

    std::unique_ptr<Decoder<float>> inputDecoder  = MakeDecoder<float>(GetTensorInfo(m_Data.m_Inputs[0]),
                                                                       m_Data.m_Inputs[0]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(m_Data.m_Outputs[0]),
                                                                       m_Data.m_Outputs[0]->Map());

    InstanceNorm(m_Data, *inputDecoder, *outputEncoder);
}

} // namespace armnn
