//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefPreluWorkload.hpp"

#include "RefWorkloadUtils.hpp"
#include "PreluImpl.hpp"

#include <Profiling.hpp>

namespace armnn
{

RefPreluWorkload::RefPreluWorkload(const PreluQueueDescriptor& descriptor,
                                   const WorkloadInfo& info)
    : BaseWorkload(descriptor, info)
{}

void RefPreluWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefPreluWorkload_Execute");

    std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(GetTensorInfo(m_Data.m_Inputs[0]),
                                                                      m_Data.m_Inputs[0]->Map());
    std::unique_ptr<Decoder<float>> alphaDecoder = MakeDecoder<float>(GetTensorInfo(m_Data.m_Inputs[1]),
                                                                      m_Data.m_Inputs[1]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(m_Data.m_Outputs[0]),
                                                                       m_Data.m_Outputs[0]->Map());

    PreluImpl(m_Data, *inputDecoder, *alphaDecoder, *outputEncoder);
}

} // namespace armnn
