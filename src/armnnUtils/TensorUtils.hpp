//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/TypesUtils.hpp>

namespace armnnUtils
{
armnn::TensorShape GetTensorShape(unsigned int numberOfBatches,
                                  unsigned int numberOfChannels,
                                  unsigned int height,
                                  unsigned int width,
                                  const armnn::DataLayout dataLayout);

template<typename T>
armnn::TensorInfo GetTensorInfo(unsigned int numberOfBatches,
                                unsigned int numberOfChannels,
                                unsigned int height,
                                unsigned int width,
                                const armnn::DataLayout dataLayout)
{
    switch (dataLayout)
    {
        case armnn::DataLayout::NCHW:
            return armnn::TensorInfo({numberOfBatches, numberOfChannels, height, width}, armnn::GetDataType<T>());
        case armnn::DataLayout::NHWC:
            return armnn::TensorInfo({numberOfBatches, height, width, numberOfChannels}, armnn::GetDataType<T>());
        default:
            throw armnn::InvalidArgumentException("Unknown data layout ["
                                                  + std::to_string(static_cast<int>(dataLayout)) +
                                                  "]", CHECK_LOCATION());
    }
}
} // namespace armnnUtils