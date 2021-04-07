//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefCastWorkload.hpp"
#include "RefWorkloadUtils.hpp"
#include <armnnUtils/FloatingPointConverter.hpp>
#include <ResolveType.hpp>
#include "Encoders.hpp"
#include "Decoders.hpp"

namespace
{
    void Cast(armnn::Decoder<float>& in, armnn::Encoder<float>& out, const uint32_t numElements )
    {
        for (unsigned int i = 0; i < numElements; i++)
        {
            out.Set(in.Get());
            ++in;
            ++out;
        }
    }
}

namespace armnn
{

    void RefCastWorkload::Execute() const
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefCastWorkload_Execute");
        const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
        const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

        Cast(*MakeDecoder<float>(inputInfo, m_Data.m_Inputs[0]->Map()),
             *MakeEncoder<float>(outputInfo, m_Data.m_Outputs[0]->Map()),
             inputInfo.GetNumElements());
    }

} //namespace armnn