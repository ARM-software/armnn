//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefComparisonWorkload.hpp"
#include "ElementwiseFunction.hpp"
#include "RefWorkloadUtils.hpp"
#include "Profiling.hpp"
#include <vector>

namespace armnn {

template <typename Functor, typename ParentDescriptor, typename armnn::StringMapping::Id DebugString>
void RefComparisonWorkload<Functor, ParentDescriptor, DebugString>::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, StringMapping::Instance().Get(DebugString));
    const TensorInfo& inputInfo0 = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(m_Data.m_Inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    const TensorShape& inShape0 = inputInfo0.GetShape();
    const TensorShape& inShape1 = inputInfo1.GetShape();
    const TensorShape& outShape = outputInfo.GetShape();

    switch(inputInfo0.GetDataType())
    {
        case armnn::DataType::QuantisedAsymm8:
        {
            QASymm8Decoder decodeIterator0(GetInputTensorDataU8(0, m_Data),
                                           inputInfo0.GetQuantizationScale(),
                                           inputInfo0.GetQuantizationOffset());

            QASymm8Decoder decodeIterator1(GetInputTensorDataU8(1, m_Data),
                                           inputInfo1.GetQuantizationScale(),
                                           inputInfo1.GetQuantizationOffset());

            BooleanEncoder encodeIterator0(GetOutputTensorDataU8(0, m_Data));

            ElementwiseFunction<Functor, Decoder, ComparisonEncoder>(inShape0,
                                                                     inShape1,
                                                                     outShape,
                                                                     decodeIterator0,
                                                                     decodeIterator1,
                                                                     encodeIterator0);
            break;
        }
        case armnn::DataType::Float32:
        {
            FloatDecoder decodeIterator0(GetInputTensorDataFloat(0, m_Data));
            FloatDecoder decodeIterator1(GetInputTensorDataFloat(1, m_Data));
            BooleanEncoder encodeIterator0(GetOutputTensorDataU8(0, m_Data));

            ElementwiseFunction<Functor, Decoder, ComparisonEncoder>(inShape0,
                                                                     inShape1,
                                                                     outShape,
                                                                     decodeIterator0,
                                                                     decodeIterator1,
                                                                     encodeIterator0);
            break;
        }
        default:
            BOOST_ASSERT_MSG(false, "RefComparisonWorkload: Not supported Data Type!");
            break;
    }
}

}

template class armnn::RefComparisonWorkload<std::equal_to<float>,
                                           armnn::EqualQueueDescriptor,
                                           armnn::StringMapping::RefEqualWorkload_Execute>;

template class armnn::RefComparisonWorkload<std::greater<float>,
                                           armnn::GreaterQueueDescriptor,
                                           armnn::StringMapping::RefGreaterWorkload_Execute>;
