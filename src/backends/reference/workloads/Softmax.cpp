//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Softmax.hpp"

#include <cmath>
#include <vector>

namespace armnn
{

unsigned int GetNumElementsBetween(const TensorShape& shape,
                                   unsigned int firstAxisInclusive,
                                   unsigned int lastAxisExclusive)
{
    BOOST_ASSERT(0 <= firstAxisInclusive);
    BOOST_ASSERT(firstAxisInclusive <= lastAxisExclusive);
    BOOST_ASSERT(lastAxisExclusive <= shape.GetNumDimensions());
    unsigned int count = 1;
    for (unsigned int i = firstAxisInclusive; i < lastAxisExclusive; i++)
    {
        count *= shape[i];
    }
    return count;
}

/// Computes the softmax function on some inputs, into outputs, with a shape given by tensorInfo.
void Softmax(Decoder<float>& in, Encoder<float>& out, const TensorInfo& inputTensorInfo, float beta, int axis)
{
    BOOST_ASSERT_MSG(axis < static_cast<int>(inputTensorInfo.GetNumDimensions()),
                     "Required axis index greater than number of dimensions.");
    BOOST_ASSERT_MSG(axis >= -static_cast<int>(inputTensorInfo.GetNumDimensions()),
                     "Required axis index lower than negative of the number of dimensions");

    unsigned int uAxis = axis < 0  ?
                         inputTensorInfo.GetNumDimensions() - static_cast<unsigned int>(abs(axis))
                         : static_cast<unsigned int>(axis);

    const TensorShape& inputShape = inputTensorInfo.GetShape();
    const unsigned int outerSize  = GetNumElementsBetween(inputShape, 0, uAxis);
    const unsigned int axisSize   = inputShape[uAxis];
    const unsigned int innerSize  = GetNumElementsBetween(inputShape, uAxis + 1, inputShape.GetNumDimensions());

    for (unsigned int outer = 0; outer < outerSize; ++outer)
    {
        unsigned int inputBeginIdx  = outer * axisSize * innerSize;
        unsigned int inputEndIdx    = inputBeginIdx + axisSize * innerSize;
        unsigned int outputBeginIdx = outer * axisSize * innerSize;

        for (unsigned int inner = 0; inner < innerSize; ++inner, ++inputBeginIdx, ++inputEndIdx, ++outputBeginIdx)
        {
            // Find max
            float maxValue = std::numeric_limits<float>::lowest();
            for (unsigned int iter = inputBeginIdx; iter < inputEndIdx; iter += innerSize)
            {
                in[iter];
                maxValue = std::max(maxValue, in.Get());
            }

            // Compute sum
            float sum = 0.0f;
            for (unsigned int iter = inputBeginIdx; iter < inputEndIdx; iter += innerSize)
            {
                in[iter];
                sum += std::exp((in.Get() - maxValue) * beta);
            }

            // Compute result
            unsigned int outputIter = outputBeginIdx;
            out[outputIter];
            for (unsigned int iter = inputBeginIdx; iter < inputEndIdx; iter += innerSize, outputIter += innerSize)
            {
                out[outputIter];
                in[iter];
                out.Set(std::exp((in.Get() - maxValue) * beta) / sum);
            }
        }
    }
}

} //namespace armnn
