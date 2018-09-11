//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ArithmeticFunction.hpp"
#include "Broadcast.hpp"
#include <functional>

namespace armnn
{

template <typename Functor>
ArithmeticFunction<Functor>::ArithmeticFunction(const TensorShape& inShape0,
                                                const TensorShape& inShape1,
                                                const TensorShape& outShape,
                                                const float* inData0,
                                                const float* inData1,
                                                float* outData)
{
    BroadcastLoop(inShape0, inShape1, outShape).Unroll(Functor(), 0, inData0, inData1, outData);
}

} //namespace armnn

template struct armnn::ArithmeticFunction<std::plus<float>>;
template struct armnn::ArithmeticFunction<std::minus<float>>;
template struct armnn::ArithmeticFunction<std::multiplies<float>>;
template struct armnn::ArithmeticFunction<std::divides<float>>;
