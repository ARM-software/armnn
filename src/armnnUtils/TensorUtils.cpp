//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TensorUtils.hpp"
#include <backendsCommon/ITensorHandle.hpp>

namespace armnnUtils
{

armnn::TensorShape GetTensorShape(unsigned int numberOfBatches,
                                  unsigned int numberOfChannels,
                                  unsigned int height,
                                  unsigned int width,
                                  const armnn::DataLayout dataLayout)
{
    switch (dataLayout)
    {
        case armnn::DataLayout::NCHW:
            return armnn::TensorShape({numberOfBatches, numberOfChannels, height, width});
        case armnn::DataLayout::NHWC:
            return armnn::TensorShape({numberOfBatches, height, width, numberOfChannels});
        default:
            throw armnn::InvalidArgumentException("Unknown data layout ["
                                                  + std::to_string(static_cast<int>(dataLayout)) +
                                                  "]", CHECK_LOCATION());
    }
}

armnn::TensorInfo GetTensorInfo(unsigned int numberOfBatches,
                                unsigned int numberOfChannels,
                                unsigned int height,
                                unsigned int width,
                                const armnn::DataLayout dataLayout,
                                const armnn::DataType dataType)
{
    switch (dataLayout)
    {
        case armnn::DataLayout::NCHW:
            return armnn::TensorInfo({numberOfBatches, numberOfChannels, height, width}, dataType);
        case armnn::DataLayout::NHWC:
            return armnn::TensorInfo({numberOfBatches, height, width, numberOfChannels}, dataType);
        default:
            throw armnn::InvalidArgumentException("Unknown data layout ["
                                                  + std::to_string(static_cast<int>(dataLayout)) +
                                                  "]", CHECK_LOCATION());
    }
}

std::pair<float, float> FindMinMax(armnn::ITensorHandle* tensorHandle)
{
    auto tensor_data = static_cast<const float *>(tensorHandle->Map(true));
    auto tensor_size = tensorHandle->GetShape().GetNumElements();

    // Set min/max initially to first value in tensor
    float min = tensor_data[0];
    float max = tensor_data[0];

    // Loop over rest of tensor and update min/max if necessary
    for (unsigned int val = 1; val < tensor_size; val++)
    {
        if (tensor_data[val] < min)
        {
            min = tensor_data[val];
        }
        else if (tensor_data[val] > max)
        {
            max = tensor_data[val];
        }
    }

    tensorHandle->Unmap();

    return std::make_pair(min, max);
}

}
