//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefElementwiseWorkload.hpp"
#include "ElementwiseFunction.hpp"
#include "RefWorkloadUtils.hpp"
#include "Profiling.hpp"
#include <vector>

namespace armnn
{

template <typename ParentDescriptor, typename Functor>
void BaseFloat32ElementwiseWorkload<ParentDescriptor, Functor>::ExecuteImpl(const char * debugString) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, debugString);

    auto data = Float32Workload<ParentDescriptor>::GetData();
    const TensorShape& inShape0 = GetTensorInfo(data.m_Inputs[0]).GetShape();
    const TensorShape& inShape1 = GetTensorInfo(data.m_Inputs[1]).GetShape();
    const TensorShape& outShape = GetTensorInfo(data.m_Outputs[0]).GetShape();

    const float* inData0 = GetInputTensorDataFloat(0, data);
    const float* inData1 = GetInputTensorDataFloat(1, data);
    float* outData = GetOutputTensorDataFloat(0, data);

    ElementwiseFunction<Functor, float, float>(inShape0, inShape1, outShape, inData0, inData1, outData);
}

template <typename ParentDescriptor, typename Functor>
void BaseUint8ElementwiseWorkload<ParentDescriptor, Functor>::ExecuteImpl(const char * debugString) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, debugString);

    auto data = Uint8Workload<ParentDescriptor>::GetData();
    const TensorInfo& inputInfo0 = GetTensorInfo(data.m_Inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(data.m_Inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(data.m_Outputs[0]);

    auto dequant0 = Dequantize(GetInputTensorDataU8(0, data), inputInfo0);
    auto dequant1 = Dequantize(GetInputTensorDataU8(1, data), inputInfo1);

    std::vector<float> results(outputInfo.GetNumElements());

    ElementwiseFunction<Functor, float, float>(inputInfo0.GetShape(),
                                               inputInfo1.GetShape(),
                                               outputInfo.GetShape(),
                                               dequant0.data(),
                                               dequant1.data(),
                                               results.data());

    Quantize(GetOutputTensorDataU8(0, data), results.data(), outputInfo);
}

}

template class armnn::BaseFloat32ElementwiseWorkload<armnn::AdditionQueueDescriptor, std::plus<float>>;
template class armnn::BaseUint8ElementwiseWorkload<armnn::AdditionQueueDescriptor, std::plus<float>>;

template class armnn::BaseFloat32ElementwiseWorkload<armnn::SubtractionQueueDescriptor, std::minus<float>>;
template class armnn::BaseUint8ElementwiseWorkload<armnn::SubtractionQueueDescriptor, std::minus<float>>;

template class armnn::BaseFloat32ElementwiseWorkload<armnn::MultiplicationQueueDescriptor, std::multiplies<float>>;
template class armnn::BaseUint8ElementwiseWorkload<armnn::MultiplicationQueueDescriptor, std::multiplies<float>>;

template class armnn::BaseFloat32ElementwiseWorkload<armnn::DivisionQueueDescriptor, std::divides<float>>;
template class armnn::BaseUint8ElementwiseWorkload<armnn::DivisionQueueDescriptor, std::divides<float>>;

template class armnn::BaseFloat32ElementwiseWorkload<armnn::MaximumQueueDescriptor, armnn::maximum<float>>;
template class armnn::BaseUint8ElementwiseWorkload<armnn::MaximumQueueDescriptor, armnn::maximum<float>>;

template class armnn::BaseFloat32ElementwiseWorkload<armnn::MinimumQueueDescriptor, armnn::minimum<float>>;
template class armnn::BaseUint8ElementwiseWorkload<armnn::MinimumQueueDescriptor, armnn::minimum<float>>;
