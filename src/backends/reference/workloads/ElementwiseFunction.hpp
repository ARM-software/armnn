//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"
#include <armnn/Tensor.hpp>

namespace armnn
{

template <typename Functor, typename DecoderOp, typename EncoderOp>
struct ElementwiseFunction
{
    ElementwiseFunction(const TensorShape& inShape0,
                        const TensorShape& inShape1,
                        const TensorShape& outShape,
                        DecoderOp& inData0,
                        DecoderOp& inData1,
                        EncoderOp& outData);
};

} //namespace armnn
