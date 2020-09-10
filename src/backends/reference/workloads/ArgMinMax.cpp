//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ArgMinMax.hpp"

#include <armnnUtils/TensorUtils.hpp>

#include <armnn/utility/NumericCast.hpp>

namespace armnn
{

template <typename OUT>
void ArgMinMax(Decoder<float>& in, OUT* out, const TensorInfo& inputTensorInfo,
               const TensorInfo& outputTensorInfo, ArgMinMaxFunction function, int axis)
{
    IgnoreUnused(outputTensorInfo);

    unsigned int uAxis = armnnUtils::GetUnsignedAxis(inputTensorInfo.GetNumDimensions(), axis);

    const unsigned int outerElements = armnnUtils::GetNumElementsBetween(inputTensorInfo.GetShape(), 0, uAxis);
    const unsigned int axisSize = inputTensorInfo.GetShape()[uAxis];
    const unsigned int innerElements = armnnUtils::GetNumElementsBetween(inputTensorInfo.GetShape(),
                                                                         uAxis + 1,
                                                                         inputTensorInfo.GetNumDimensions());

    for (unsigned int outer = 0; outer < outerElements; ++outer) {
        for (unsigned int inner = 0; inner < innerElements; ++inner) {
            in[outer * axisSize * innerElements + inner];
            auto tmpValue = in.Get();
            unsigned int tmpIndex = 0;
            for (unsigned int i = 1; i < axisSize; ++i) {
                in[(outer * axisSize * innerElements) + (i * innerElements) + inner];
                const auto& value = in.Get();
                if ((function == armnn::ArgMinMaxFunction::Min && value < tmpValue) ||
                    (function == armnn::ArgMinMaxFunction::Max &&  value > tmpValue)) {
                    tmpValue = value;
                    tmpIndex = i;
                }
            }

            out[outer * innerElements + inner] = armnn::numeric_cast<OUT>(tmpIndex);
        }
    }
}

template void ArgMinMax(Decoder<float>& in, int32_t* out, const TensorInfo& inputTensorInfo,
               const TensorInfo& outputTensorInfo, ArgMinMaxFunction function, int axis);

template void ArgMinMax(Decoder<float>& in, int64_t* out, const TensorInfo& inputTensorInfo,
               const TensorInfo& outputTensorInfo, ArgMinMaxFunction function, int axis);

} //namespace armnn
