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
struct ElementwiseBinaryFunction
{
    using OutType = typename Functor::result_type;
    using InType = typename Functor::first_argument_type;

    ElementwiseBinaryFunction(const TensorShape& inShape0,
                              const TensorShape& inShape1,
                              const TensorShape& outShape,
                              Decoder<InType>& inData0,
                              Decoder<InType>& inData1,
                              Encoder<OutType>& outData);
};

template <typename Functor>
struct ElementwiseUnaryFunction
{
    using OutType = typename Functor::result_type;
    using InType = typename Functor::argument_type;

    ElementwiseUnaryFunction(const TensorShape& inShape,
                             const TensorShape& outShape,
                             Decoder<InType>& inData,
                             Encoder<OutType>& outData);
};

template <typename Functor>
struct LogicalBinaryFunction
{
    using OutType = bool;
    using InType = bool;

    LogicalBinaryFunction(const TensorShape& inShape0,
                          const TensorShape& inShape1,
                          const TensorShape& outShape,
                          Decoder<InType>& inData0,
                          Decoder<InType>& inData1,
                          Encoder<OutType>& outData);
};

template <typename Functor>
struct LogicalUnaryFunction
{
    using OutType = bool;
    using InType = bool;

    LogicalUnaryFunction(const TensorShape& inShape,
                         const TensorShape& outShape,
                         Decoder<InType>& inData,
                         Encoder<OutType>& outData);
};

} //namespace armnn
