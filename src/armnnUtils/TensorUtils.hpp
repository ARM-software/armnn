//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/TypesUtils.hpp>

#include <boost/assert.hpp>

namespace armnnUtils
{
armnn::TensorShape GetTensorShape(unsigned int numberOfBatches,
                                  unsigned int numberOfChannels,
                                  unsigned int height,
                                  unsigned int width,
                                  const armnn::DataLayout dataLayout);

armnn::TensorInfo GetTensorInfo(unsigned int numberOfBatches,
                                unsigned int numberOfChannels,
                                unsigned int height,
                                unsigned int width,
                                const armnn::DataLayout dataLayout,
                                const armnn::DataType dataType);

std::pair<float, float> FindMinMax(armnn::ITensorHandle* tensorHandle);

armnn::TensorShape ExpandDims(const armnn::TensorShape& tensorShape, int axis);

unsigned int GetNumElementsBetween(const armnn::TensorShape& shape,
                                   unsigned int firstAxisInclusive,
                                   unsigned int lastAxisExclusive);

unsigned int GetUnsignedAxis(const unsigned int inputDimension, const int axis);

inline unsigned int GetNumElementsAfter(const armnn::TensorShape& shape,
                                        unsigned int axis)
{
    unsigned int numDim = shape.GetNumDimensions();
    BOOST_ASSERT(0 >= axis);
    BOOST_ASSERT(axis < numDim - 1);
    unsigned int count = 1;
    for (unsigned int i = axis; i < numDim; i++)
    {
        count *= shape[i];
    }
    return count;
}

inline std::pair<unsigned int, std::vector<float>> GetPerAxisParams(const armnn::TensorInfo& info)
{
    const std::vector<float>& scales = info.GetQuantizationScales();
    armnn::Optional<unsigned int> quantizationDim = info.GetQuantizationDim();
    if (scales.size() < 1 || !quantizationDim.has_value())
    {
        throw armnn::InvalidArgumentException(
        "We currently support only per-axis symmetric quantization for QuantizedSymm8.");
    }
    unsigned int axisFactor = GetNumElementsAfter(info.GetShape(), quantizationDim.value());

    return {axisFactor, scales};
}

} // namespace armnnUtils
