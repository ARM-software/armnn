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

template <typename Functor,
          typename armnn::DataType DataType,
          typename ParentDescriptor,
          typename armnn::StringMapping::Id DebugString>
void RefElementwiseWorkload<Functor, DataType, ParentDescriptor, DebugString>::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, StringMapping::Instance().Get(DebugString));

    const TensorInfo& inputInfo0 = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(m_Data.m_Inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    const TensorShape& inShape0 = inputInfo0.GetShape();
    const TensorShape& inShape1 = inputInfo1.GetShape();
    const TensorShape& outShape = outputInfo.GetShape();

    switch(DataType)
    {
        case armnn::DataType::QuantisedAsymm8:
        {
            std::vector<float> results(outputInfo.GetNumElements());
            ElementwiseFunction<Functor, float, float>(inShape0,
                                                       inShape1,
                                                       outShape,
                                                       Dequantize(GetInputTensorDataU8(0, m_Data), inputInfo0).data(),
                                                       Dequantize(GetInputTensorDataU8(1, m_Data), inputInfo1).data(),
                                                       results.data());
            Quantize(GetOutputTensorDataU8(0, m_Data), results.data(), outputInfo);
            break;
        }
        case armnn::DataType::Float32:
        {
            ElementwiseFunction<Functor, float, float>(inShape0,
                                                       inShape1,
                                                       outShape,
                                                       GetInputTensorDataFloat(0, m_Data),
                                                       GetInputTensorDataFloat(1, m_Data),
                                                       GetOutputTensorDataFloat(0, m_Data));
            break;
        }
        default:
            BOOST_ASSERT_MSG(false, "Unknown Data Type!");
            break;
    }
}

}

template class armnn::RefElementwiseWorkload<std::plus<float>,
    armnn::DataType::Float32,
    armnn::AdditionQueueDescriptor,
    armnn::StringMapping::RefAdditionWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::plus<float>,
    armnn::DataType::QuantisedAsymm8,
    armnn::AdditionQueueDescriptor,
    armnn::StringMapping::RefAdditionWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::minus<float>,
    armnn::DataType::Float32,
    armnn::SubtractionQueueDescriptor,
    armnn::StringMapping::RefSubtractionWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::minus<float>,
    armnn::DataType::QuantisedAsymm8,
    armnn::SubtractionQueueDescriptor,
    armnn::StringMapping::RefSubtractionWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::multiplies<float>,
    armnn::DataType::Float32,
    armnn::MultiplicationQueueDescriptor,
    armnn::StringMapping::RefMultiplicationWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::multiplies<float>,
    armnn::DataType::QuantisedAsymm8,
    armnn::MultiplicationQueueDescriptor,
    armnn::StringMapping::RefMultiplicationWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::divides<float>,
    armnn::DataType::Float32,
    armnn::DivisionQueueDescriptor,
    armnn::StringMapping::RefDivisionWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::divides<float>,
    armnn::DataType::QuantisedAsymm8,
    armnn::DivisionQueueDescriptor,
    armnn::StringMapping::RefDivisionWorkload_Execute>;

template class armnn::RefElementwiseWorkload<armnn::maximum<float>,
    armnn::DataType::Float32,
    armnn::MaximumQueueDescriptor,
    armnn::StringMapping::RefMaximumWorkload_Execute>;

template class armnn::RefElementwiseWorkload<armnn::maximum<float>,
    armnn::DataType::QuantisedAsymm8,
    armnn::MaximumQueueDescriptor,
    armnn::StringMapping::RefMaximumWorkload_Execute>;


template class armnn::RefElementwiseWorkload<armnn::minimum<float>,
    armnn::DataType::Float32,
    armnn::MinimumQueueDescriptor,
    armnn::StringMapping::RefMinimumWorkload_Execute>;

template class armnn::RefElementwiseWorkload<armnn::minimum<float>,
    armnn::DataType::QuantisedAsymm8,
    armnn::MinimumQueueDescriptor,
    armnn::StringMapping::RefMinimumWorkload_Execute>;
