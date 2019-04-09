//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"
#include <armnn/Tensor.hpp>

namespace armnn
{

template <typename Functor>
struct ElementwiseFunction
{
    using OutType = typename Functor::result_type;
    using InType = typename Functor::first_argument_type;

    ElementwiseFunction(const TensorShape& inShape0,
                        const TensorShape& inShape1,
                        const TensorShape& outShape,
                        armnn::Decoder<InType>& inData0,
                        armnn::Decoder<InType>& inData1,
                        armnn::Encoder<OutType>& outData);
};

} //namespace armnn
