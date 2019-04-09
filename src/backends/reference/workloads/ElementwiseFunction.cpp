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

template <typename Functor>
ElementwiseFunction<Functor>::ElementwiseFunction(const TensorShape& inShape0,
                                                   const TensorShape& inShape1,
                                                   const TensorShape& outShape,
                                                   armnn::Decoder<InType>& inData0,
                                                   armnn::Decoder<InType>& inData1,
                                                   armnn::Encoder<OutType>& outData)
{
    BroadcastLoop(inShape0, inShape1, outShape).Unroll(Functor(), 0, inData0, inData1, outData);
}

} //namespace armnn

template struct armnn::ElementwiseFunction<std::plus<float>>;
template struct armnn::ElementwiseFunction<std::minus<float>>;
template struct armnn::ElementwiseFunction<std::multiplies<float>>;
template struct armnn::ElementwiseFunction<std::divides<float>>;
template struct armnn::ElementwiseFunction<armnn::maximum<float>>;
template struct armnn::ElementwiseFunction<armnn::minimum<float>>;
template struct armnn::ElementwiseFunction<std::equal_to<float>>;
template struct armnn::ElementwiseFunction<std::greater<float>>;

