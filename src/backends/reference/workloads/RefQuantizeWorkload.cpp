//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefQuantizeWorkload.hpp"

#include <armnn/TypesUtils.hpp>


namespace armnn
{

namespace
{

template<typename T>
void QuantizeImpl(const void *input, void *output, size_t numValues, float scale, int offset)
{
    auto in = static_cast<const float *>(input);
    auto out = static_cast<T *>(output);
    for (size_t i = 0; i < numValues; i++, in++, out++)
    {
        *out = armnn::Quantize<T>(*in, scale, offset);
    }
}

} //namespace

RefQuantizeWorkload::RefQuantizeWorkload(const QuantizeQueueDescriptor& descriptor, const WorkloadInfo &info)
    : BaseWorkload(descriptor, info)
    , m_NumElements(info.m_InputTensorInfos[0].GetNumElements())
    , m_TargetType(info.m_OutputTensorInfos[0].GetDataType())
    , m_Scale(info.m_OutputTensorInfos[0].GetQuantizationScale())
    , m_Offset(info.m_OutputTensorInfos[0].GetQuantizationOffset())
{
}

void RefQuantizeWorkload::Execute() const
{
    const void* input = m_Data.m_Inputs[0]->Map(true);
    void* output =  m_Data.m_Outputs[0]->Map(true);

    switch(m_TargetType)
    {
        case DataType::QuantisedAsymm8:
        {
            QuantizeImpl<uint8_t>(input, output, m_NumElements, m_Scale, m_Offset);
            break;
        }
        case DataType::QuantisedSymm16:
        {
            QuantizeImpl<int16_t>(input, output, m_NumElements, m_Scale, 0);
            break;
        }
        default:
        {
            BOOST_ASSERT_MSG(false, "RefQuantizeWorkload: Non quantized output type encountered");
        }
    }

    m_Data.m_Inputs[0]->Unmap();
    m_Data.m_Outputs[0]->Unmap();
}

} //namespace armnn