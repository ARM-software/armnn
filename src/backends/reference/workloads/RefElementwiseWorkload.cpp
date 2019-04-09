//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefElementwiseWorkload.hpp"
#include "ElementwiseFunction.hpp"
#include "RefWorkloadUtils.hpp"
#include "Profiling.hpp"
#include "StringMapping.hpp"
#include "TypeUtils.hpp"
#include <vector>

namespace armnn
{

template <typename Functor, typename ParentDescriptor, typename armnn::StringMapping::Id DebugString>
void RefElementwiseWorkload<Functor, ParentDescriptor, DebugString>::Execute() const
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

            QASymm8Encoder encodeIterator0(GetOutputTensorDataU8(0, m_Data),
                                           outputInfo.GetQuantizationScale(),
                                           outputInfo.GetQuantizationOffset());

            ElementwiseFunction<Functor, Decoder, Encoder>(inShape0,
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
            FloatEncoder encodeIterator0(GetOutputTensorDataFloat(0, m_Data));

            ElementwiseFunction<Functor, Decoder, Encoder>(inShape0,
                                                           inShape1,
                                                           outShape,
                                                           decodeIterator0,
                                                           decodeIterator1,
                                                           encodeIterator0);
            break;
        }
        case armnn::DataType::QuantisedSymm16:
        {
            QSymm16Decoder decodeIterator0(GetInputTensorData<int16_t>(0, m_Data),
                                           inputInfo0.GetQuantizationScale(),
                                           inputInfo0.GetQuantizationOffset());

            QSymm16Decoder decodeIterator1(GetInputTensorData<int16_t>(1, m_Data),
                                           inputInfo1.GetQuantizationScale(),
                                           inputInfo1.GetQuantizationOffset());

            QSymm16Encoder encodeIterator0(GetOutputTensorData<int16_t>(0, m_Data),
                                           outputInfo.GetQuantizationScale(),
                                           outputInfo.GetQuantizationOffset());

            ElementwiseFunction<Functor, Decoder, Encoder>(inShape0,
                                                           inShape1,
                                                           outShape,
                                                           decodeIterator0,
                                                           decodeIterator1,
                                                           encodeIterator0);
            break;
        }
        default:
            BOOST_ASSERT_MSG(false, "RefElementwiseWorkload: Not supported Data Type!");
            break;
    }
}

}

template class armnn::RefElementwiseWorkload<std::plus<float>,
                                            armnn::AdditionQueueDescriptor,
                                            armnn::StringMapping::RefAdditionWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::minus<float>,
                                            armnn::SubtractionQueueDescriptor,
                                            armnn::StringMapping::RefSubtractionWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::multiplies<float>,
                                            armnn::MultiplicationQueueDescriptor,
                                            armnn::StringMapping::RefMultiplicationWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::divides<float>,
                                            armnn::DivisionQueueDescriptor,
                                            armnn::StringMapping::RefDivisionWorkload_Execute>;

template class armnn::RefElementwiseWorkload<armnn::maximum<float>,
                                            armnn::MaximumQueueDescriptor,
                                            armnn::StringMapping::RefMaximumWorkload_Execute>;

template class armnn::RefElementwiseWorkload<armnn::minimum<float>,
                                            armnn::MinimumQueueDescriptor,
                                            armnn::StringMapping::RefMinimumWorkload_Execute>;