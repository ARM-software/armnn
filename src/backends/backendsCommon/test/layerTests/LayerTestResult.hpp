//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/utility/Assert.hpp>

#include <boost/multi_array.hpp>

#include <cstddef>

template <std::size_t n>
boost::array<unsigned int, n> GetTensorShapeAsArray(const armnn::TensorInfo& tensorInfo)
{
    ARMNN_ASSERT_MSG(n == tensorInfo.GetNumDimensions(),
        "Attempting to construct a shape array of mismatching size");

    boost::array<unsigned int, n> shape;
    for (unsigned int i = 0; i < n; i++)
    {
        shape[i] = tensorInfo.GetShape()[i];
    }
    return shape;
}

template <typename T, std::size_t n>
struct LayerTestResult
{
    LayerTestResult(const armnn::TensorInfo& outputInfo)
    {
        auto shape( GetTensorShapeAsArray<n>(outputInfo) );
        output.resize(shape);
        outputExpected.resize(shape);
        supported = true;
        compareBoolean = false;
    }

    boost::multi_array<T, n> output;
    boost::multi_array<T, n> outputExpected;
    bool supported;
    bool compareBoolean;
};
