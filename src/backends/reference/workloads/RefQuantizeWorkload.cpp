//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefQuantizeWorkload.hpp"

#include "RefWorkloadUtils.hpp"

#include <armnn/TypesUtils.hpp>


namespace armnn
{

namespace
{

void QuantizeImpl(Decoder<float>& in, Encoder<float>& out, size_t numValues)
{
    for (unsigned int i = 0; i < numValues; i++)
    {
        in[i];
        out[i];
        out.Set(in.Get());
    }
}

} //namespace

RefQuantizeWorkload::RefQuantizeWorkload(const QuantizeQueueDescriptor& descriptor, const WorkloadInfo &info)
    : BaseWorkload(descriptor, info)
    , m_NumElements(info.m_InputTensorInfos[0].GetNumElements())
{
}

void RefQuantizeWorkload::PostAllocationConfigure()
{
    const TensorInfo& inputInfo = armnn::GetTensorInfo(m_Data.m_Inputs[0]);
    m_InputDecoder = MakeDecoder<float>(inputInfo);

    const TensorInfo& outputInfo = armnn::GetTensorInfo(m_Data.m_Outputs[0]);
    m_OutputEncoder = MakeEncoder<float>(outputInfo);
}

void RefQuantizeWorkload::Execute() const
{
    m_InputDecoder->Reset(m_Data.m_Inputs[0]->Map());
    m_OutputEncoder->Reset(m_Data.m_Outputs[0]->Map());

    QuantizeImpl(*m_InputDecoder, *m_OutputEncoder, m_NumElements);
}

} //namespace armnn