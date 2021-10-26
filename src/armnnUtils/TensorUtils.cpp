//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnnUtils/TensorUtils.hpp>

#include <armnn/backends/ITensorHandle.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <fmt/format.h>

using namespace armnn;

namespace armnnUtils
{

TensorShape GetTensorShape(unsigned int numberOfBatches,
                                  unsigned int numberOfChannels,
                                  unsigned int height,
                                  unsigned int width,
                                  const DataLayout dataLayout)
{
    switch (dataLayout)
    {
        case DataLayout::NCHW:
            return TensorShape({numberOfBatches, numberOfChannels, height, width});
        case DataLayout::NHWC:
            return TensorShape({numberOfBatches, height, width, numberOfChannels});
        default:
            throw InvalidArgumentException("Unknown data layout ["
                                                  + std::to_string(static_cast<int>(dataLayout)) +
                                                  "]", CHECK_LOCATION());
    }
}

TensorInfo GetTensorInfo(unsigned int numberOfBatches,
                                unsigned int numberOfChannels,
                                unsigned int height,
                                unsigned int width,
                                const DataLayout dataLayout,
                                const DataType dataType)
{
    switch (dataLayout)
    {
        case DataLayout::NCHW:
            return TensorInfo({numberOfBatches, numberOfChannels, height, width}, dataType);
        case DataLayout::NHWC:
            return TensorInfo({numberOfBatches, height, width, numberOfChannels}, dataType);
        default:
            throw InvalidArgumentException("Unknown data layout ["
                                                  + std::to_string(static_cast<int>(dataLayout)) +
                                                  "]", CHECK_LOCATION());
    }
}

TensorInfo GetTensorInfo(unsigned int numberOfBatches,
                                unsigned int numberOfChannels,
                                unsigned int depth,
                                unsigned int height,
                                unsigned int width,
                                const DataLayout dataLayout,
                                const DataType dataType)
{
    switch (dataLayout)
    {
        case DataLayout::NDHWC:
            return TensorInfo({numberOfBatches, depth, height, width, numberOfChannels}, dataType);
        case DataLayout::NCDHW:
            return TensorInfo({numberOfBatches, numberOfChannels, depth, height, width}, dataType);
        default:
            throw InvalidArgumentException("Unknown data layout ["
                                                  + std::to_string(static_cast<int>(dataLayout)) +
                                                  "]", CHECK_LOCATION());
    }
}

std::pair<float, float> FindMinMax(ITensorHandle* tensorHandle)
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

TensorShape ExpandDims(const TensorShape& tensorShape, int axis)
{
    unsigned int outputDim = tensorShape.GetNumDimensions() + 1;

    if (axis < -armnn::numeric_cast<int>(outputDim) || axis > armnn::numeric_cast<int>(tensorShape.GetNumDimensions()))
    {
        throw InvalidArgumentException(fmt::format("Invalid expansion axis {} for {}D input tensor. {}",
                                                   axis,
                                                   tensorShape.GetNumDimensions(),
                                                   CHECK_LOCATION().AsString()));
    }

    if (axis < 0)
    {
        axis = armnn::numeric_cast<int>(outputDim) + axis;
    }

    std::vector<unsigned int> outputShape;
    outputShape.reserve(tensorShape.GetNumDimensions());
    for (unsigned int i = 0; i < tensorShape.GetNumDimensions(); ++i)
    {
        outputShape.push_back(tensorShape[i]);
    }
    outputShape.insert(outputShape.begin() + axis, 1);

    return TensorShape(outputDim, outputShape.data());
}

unsigned int GetNumElementsBetween(const TensorShape& shape,
                                   const unsigned int firstAxisInclusive,
                                   const unsigned int lastAxisExclusive)
{
    ARMNN_ASSERT(firstAxisInclusive <= lastAxisExclusive);
    ARMNN_ASSERT(lastAxisExclusive <= shape.GetNumDimensions());
    unsigned int count = 1;
    for (unsigned int i = firstAxisInclusive; i < lastAxisExclusive; i++)
    {
        count *= shape[i];
    }
    return count;
}

unsigned int GetUnsignedAxis(const unsigned int inputDimension, const int axis)
{
    ARMNN_ASSERT_MSG(axis < armnn::numeric_cast<int>(inputDimension),
                     "Required axis index greater than number of dimensions.");
    ARMNN_ASSERT_MSG(axis >= -armnn::numeric_cast<int>(inputDimension),
                     "Required axis index lower than negative of the number of dimensions");

    unsigned int uAxis = axis < 0  ?
                         inputDimension - armnn::numeric_cast<unsigned int>(abs(axis))
                         : armnn::numeric_cast<unsigned int>(axis);
    return uAxis;
}

unsigned int GetNumElementsAfter(const armnn::TensorShape& shape, unsigned int axis)
{
    unsigned int numDim = shape.GetNumDimensions();
    ARMNN_ASSERT(axis <= numDim - 1);
    unsigned int count = 1;
    for (unsigned int i = axis+1; i < numDim; i++)
    {
        count *= shape[i];
    }
    return count;
}

std::pair<unsigned int, std::vector<float>> GetPerAxisParams(const armnn::TensorInfo& info)
{
    const std::vector<float>& scales = info.GetQuantizationScales();
    armnn::Optional<unsigned int> quantizationDim = info.GetQuantizationDim();
    if (!info.HasPerAxisQuantization())
    {
        throw armnn::InvalidArgumentException(
            std::string("Per-axis quantization params not set for tensor of type ") +
            armnn::GetDataTypeName(info.GetDataType()), CHECK_LOCATION());
    }
    unsigned int axisFactor = GetNumElementsAfter(info.GetShape(), quantizationDim.value()) ;

    return { axisFactor, scales };
}

} // namespace armnnUtils
