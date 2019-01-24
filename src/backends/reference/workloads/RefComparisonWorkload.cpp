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

template<typename ParentDescriptor, typename Functor>
void RefFloat32ComparisonWorkload<ParentDescriptor, Functor>::ExecuteImpl(const char* debugString) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, debugString);

    auto data = BaseFloat32ComparisonWorkload<ParentDescriptor>::GetData();
    const TensorShape& inShape0 = GetTensorInfo(data.m_Inputs[0]).GetShape();
    const TensorShape& inShape1 = GetTensorInfo(data.m_Inputs[1]).GetShape();
    const TensorShape& outputShape = GetTensorInfo(data.m_Outputs[0]).GetShape();

    const float* inData0 = GetInputTensorDataFloat(0, data);
    const float* inData1 = GetInputTensorDataFloat(1, data);
    uint8_t* outData = GetOutputTensorData<uint8_t>(0, data);

    ElementwiseFunction<Functor, float, uint8_t>(inShape0,
                                                 inShape1,
                                                 outputShape,
                                                 inData0,
                                                 inData1,
                                                 outData);

}

template<typename ParentDescriptor, typename Functor>
void RefUint8ComparisonWorkload<ParentDescriptor, Functor>::ExecuteImpl(const char* debugString) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, debugString);

    auto data = BaseUint8ComparisonWorkload<ParentDescriptor>::GetData();
    const TensorShape& inputInfo0 = GetTensorInfo(data.m_Inputs[0]).GetShape();
    const TensorShape& inputInfo1 = GetTensorInfo(data.m_Inputs[1]).GetShape();
    const TensorShape& outputShape = GetTensorInfo(data.m_Outputs[0]).GetShape();

    const uint8_t* inData0 = GetInputTensorData<uint8_t>(0, data);
    const uint8_t* inData1 = GetInputTensorData<uint8_t>(1, data);
    uint8_t* outData = GetOutputTensorData<uint8_t>(0, data);

    ElementwiseFunction<Functor, uint8_t, uint8_t>(inputInfo0,
                                                   inputInfo1,
                                                   outputShape,
                                                   inData0,
                                                   inData1,
                                                   outData);
}

}

template class armnn::RefFloat32ComparisonWorkload<armnn::EqualQueueDescriptor, std::equal_to<float>>;
template class armnn::RefUint8ComparisonWorkload<armnn::EqualQueueDescriptor, std::equal_to<uint8_t>>;

template class armnn::RefFloat32ComparisonWorkload<armnn::GreaterQueueDescriptor, std::greater<float>>;
template class armnn::RefUint8ComparisonWorkload<armnn::GreaterQueueDescriptor, std::greater<uint8_t>>;
