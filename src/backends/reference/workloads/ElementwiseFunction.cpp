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

template <typename Functor, typename DecoderOp, typename EncoderOp>
ElementwiseFunction<Functor, DecoderOp, EncoderOp>::ElementwiseFunction(const TensorShape& inShape0,
                                                                        const TensorShape& inShape1,
                                                                        const TensorShape& outShape,
                                                                        DecoderOp& inData0,
                                                                        DecoderOp& inData1,
                                                                        EncoderOp& outData)
{
    BroadcastLoop(inShape0, inShape1, outShape).Unroll(Functor(), 0, inData0, inData1, outData);
}

} //namespace armnn

template struct armnn::ElementwiseFunction<std::plus<float>, armnn::Decoder, armnn::Encoder>;
template struct armnn::ElementwiseFunction<std::minus<float>, armnn::Decoder, armnn::Encoder>;
template struct armnn::ElementwiseFunction<std::multiplies<float>, armnn::Decoder, armnn::Encoder>;
template struct armnn::ElementwiseFunction<std::divides<float>, armnn::Decoder, armnn::Encoder>;
template struct armnn::ElementwiseFunction<armnn::maximum<float>, armnn::Decoder, armnn::Encoder>;
template struct armnn::ElementwiseFunction<armnn::minimum<float>, armnn::Decoder, armnn::Encoder>;

template struct armnn::ElementwiseFunction<std::equal_to<float>, armnn::Decoder, armnn::ComparisonEncoder>;
template struct armnn::ElementwiseFunction<std::greater<float>, armnn::Decoder, armnn::ComparisonEncoder>;

