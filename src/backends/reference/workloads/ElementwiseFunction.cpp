//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ElementwiseFunction.hpp"
#include "Broadcast.hpp"
#include <functional>
#include "Minimum.hpp"

#include "Maximum.hpp"

namespace armnn
{

template <typename Functor, typename dataTypeInput, typename dataTypeOutput>
ElementwiseFunction<Functor, dataTypeInput, dataTypeOutput>::ElementwiseFunction(const TensorShape& inShape0,
                                                                                 const TensorShape& inShape1,
                                                                                 const TensorShape& outShape,
                                                                                 const dataTypeInput* inData0,
                                                                                 const dataTypeInput* inData1,
                                                                                 dataTypeOutput* outData)
{
    BroadcastLoop(inShape0, inShape1, outShape).Unroll(Functor(), 0, inData0, inData1, outData);
}

} //namespace armnn

template struct armnn::ElementwiseFunction<std::plus<float>, float, float>;
template struct armnn::ElementwiseFunction<std::minus<float>, float, float>;
template struct armnn::ElementwiseFunction<std::multiplies<float>, float, float>;
template struct armnn::ElementwiseFunction<std::divides<float>, float, float>;
template struct armnn::ElementwiseFunction<armnn::maximum<float>, float, float>;
template struct armnn::ElementwiseFunction<armnn::minimum<float>, float, float>;
template struct armnn::ElementwiseFunction<std::equal_to<float>, float ,uint8_t>;
template struct armnn::ElementwiseFunction<std::equal_to<uint8_t>, uint8_t, uint8_t>;
template struct armnn::ElementwiseFunction<std::greater<float>, float, uint8_t>;
template struct armnn::ElementwiseFunction<std::greater<uint8_t>, uint8_t, uint8_t>;
