//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefLogSoftmaxWorkload.hpp"

#include "Decoders.hpp"
#include "Encoders.hpp"
#include "LogSoftmax.hpp"
#include "RefWorkloadUtils.hpp"

#include <Profiling.hpp>

#include <armnn/utility/Assert.hpp>

namespace armnn
{

void RefLogSoftmaxWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefLogSoftmaxWorkload_Execute");

    const TensorInfo& inputInfo  = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    std::unique_ptr<Decoder<float>> decoder = MakeDecoder<float>(inputInfo, m_Data.m_Inputs[0]->Map());
    std::unique_ptr<Encoder<float>> encoder = MakeEncoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());

    ARMNN_ASSERT(decoder != nullptr);
    ARMNN_ASSERT(encoder != nullptr);

    LogSoftmax(*decoder, *encoder, inputInfo, m_Data.m_Parameters);
}

} // namespace armnn
