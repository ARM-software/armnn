//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LogSoftmax.hpp"

#include <armnnUtils/TensorUtils.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <cmath>

namespace
{

inline bool ValidateAxis(int axis, unsigned int numDimensions)
{
    const int sNumDimensions = armnn::numeric_cast<int>(numDimensions);
    return axis < sNumDimensions && axis >= -sNumDimensions;
}

} // anonymous namespace

namespace armnn
{

void LogSoftmax(Decoder<float>& input,
                Encoder<float>& output,
                const TensorInfo& inputInfo,
                const LogSoftmaxDescriptor& descriptor)
{
    const unsigned int numDimensions = inputInfo.GetNumDimensions();

    bool axisIsValid = ValidateAxis(descriptor.m_Axis, numDimensions);
    ARMNN_ASSERT_MSG(axisIsValid,
        "Axis index is not in range [-numDimensions, numDimensions).");
    IgnoreUnused(axisIsValid);

    unsigned int uAxis = descriptor.m_Axis < 0  ?
        numDimensions - armnn::numeric_cast<unsigned int>(std::abs(descriptor.m_Axis)) :
        armnn::numeric_cast<unsigned int>(descriptor.m_Axis);

    const TensorShape& inputShape = inputInfo.GetShape();
    const unsigned int outerSize  = armnnUtils::GetNumElementsBetween(inputShape, 0, uAxis);
    const unsigned int axisSize   = inputShape[uAxis];
    const unsigned int innerSize  = armnnUtils::GetNumElementsBetween(inputShape,
                                                                      uAxis + 1,
                                                                      inputShape.GetNumDimensions());

    for (unsigned int outer = 0; outer < outerSize; ++outer)
    {
        for (unsigned int inner = 0; inner < innerSize; ++inner)
        {
            // Find max
            input[outer * axisSize * innerSize + inner];
            float maxValue = input.Get();
            for (unsigned int i = 1u; i < axisSize; ++i)
            {
                input[(outer * axisSize + i) * innerSize + inner];
                maxValue = std::max(maxValue, input.Get());
            }

            // Compute sum
            float sum = 0.0f;
            for (unsigned int i = 0u; i < axisSize; ++i)
            {
                input[(outer * axisSize + i) * innerSize + inner];
                sum += std::exp((input.Get() - maxValue) * descriptor.m_Beta);
            }

            // Compute log sum
            const float logSum = std::log(sum);

            // Compute result
            for (unsigned int i = 0u; i < axisSize; ++i)
            {
                const unsigned int index = (outer * axisSize + i) * innerSize + inner;

                input [index];
                output[index];

                output.Set((input.Get() - maxValue) * descriptor.m_Beta - logSum);
            }
        }
    }
}

} // namespace armnn
