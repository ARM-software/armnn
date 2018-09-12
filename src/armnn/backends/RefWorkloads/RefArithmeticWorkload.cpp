//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefArithmeticWorkload.hpp"
#include "ArithmeticFunction.hpp"
#include "RefWorkloadUtils.hpp"
#include "Profiling.hpp"
#include <vector>

namespace armnn
{

template <typename ParentDescriptor, typename Functor>
void BaseFloat32ArithmeticWorkload<ParentDescriptor, Functor>::ExecuteImpl(const char * debugString) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, debugString);

    auto data = Float32Workload<ParentDescriptor>::GetData();
    const TensorShape& inShape0 = GetTensorInfo(data.m_Inputs[0]).GetShape();
    const TensorShape& inShape1 = GetTensorInfo(data.m_Inputs[1]).GetShape();
    const TensorShape& outShape = GetTensorInfo(data.m_Outputs[0]).GetShape();

    const float* inData0 = GetInputTensorDataFloat(0, data);
    const float* inData1 = GetInputTensorDataFloat(1, data);
    float* outData = GetOutputTensorDataFloat(0, data);

    ArithmeticFunction<Functor>(inShape0, inShape1, outShape, inData0, inData1, outData);
}

template <typename ParentDescriptor, typename Functor>
void BaseUint8ArithmeticWorkload<ParentDescriptor, Functor>::ExecuteImpl(const char * debugString) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, debugString);

    auto data = Uint8Workload<ParentDescriptor>::GetData();
    const TensorInfo& inputInfo0 = GetTensorInfo(data.m_Inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(data.m_Inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(data.m_Outputs[0]);

    auto dequant0 = Dequantize(GetInputTensorDataU8(0, data), inputInfo0);
    auto dequant1 = Dequantize(GetInputTensorDataU8(1, data), inputInfo1);

    std::vector<float> results(outputInfo.GetNumElements());

    ArithmeticFunction<Functor>(inputInfo0.GetShape(),
                                inputInfo1.GetShape(),
                                outputInfo.GetShape(),
                                dequant0.data(),
                                dequant1.data(),
                                results.data());

    Quantize(GetOutputTensorDataU8(0, data), results.data(), outputInfo);
}

}

template class armnn::BaseFloat32ArithmeticWorkload<armnn::AdditionQueueDescriptor, std::plus<float>>;
template class armnn::BaseUint8ArithmeticWorkload<armnn::AdditionQueueDescriptor, std::plus<float>>;

template class armnn::BaseFloat32ArithmeticWorkload<armnn::SubtractionQueueDescriptor, std::minus<float>>;
template class armnn::BaseUint8ArithmeticWorkload<armnn::SubtractionQueueDescriptor, std::minus<float>>;

template class armnn::BaseFloat32ArithmeticWorkload<armnn::MultiplicationQueueDescriptor, std::multiplies<float>>;
template class armnn::BaseUint8ArithmeticWorkload<armnn::MultiplicationQueueDescriptor, std::multiplies<float>>;

template class armnn::BaseFloat32ArithmeticWorkload<armnn::DivisionQueueDescriptor, std::divides<float>>;
template class armnn::BaseUint8ArithmeticWorkload<armnn::DivisionQueueDescriptor, std::divides<float>>;
