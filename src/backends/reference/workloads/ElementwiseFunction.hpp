//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>

namespace armnn
{

template <typename Functor, typename dataTypeInput, typename dataTypeOutput>
struct ElementwiseFunction
{
    ElementwiseFunction(const TensorShape& inShape0,
                        const TensorShape& inShape1,
                        const TensorShape& outShape,
                        const dataTypeInput* inData0,
                        const dataTypeInput* inData1,
                        dataTypeOutput* outData);
};

} //namespace armnn
