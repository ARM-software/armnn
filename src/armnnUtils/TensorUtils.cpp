//
// Copyright © 2017-2023 Arm Ltd. All rights reserved.
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

TensorShape ReduceDims(const TensorShape& tensorShape, unsigned int dimensions)
{
    if (tensorShape.GetNumDimensions() <= dimensions)
    {
        return tensorShape;
    }
    std::vector<unsigned int> newShape;

    unsigned int dimsToSkip = tensorShape.GetNumDimensions() - dimensions;
    unsigned int dimsSkipped = 0;
    bool insertRemainder = false;

    for (unsigned int i = 0; i < tensorShape.GetNumDimensions(); ++i)
    {
        if (tensorShape[i] == 1 && dimsSkipped < dimsToSkip && !insertRemainder)
        {
            ++dimsSkipped;
            continue;
        }
        newShape.push_back(tensorShape[i]);
        // Once we insert the first dimension we can't skip any more
        insertRemainder = true;
    }
    return TensorShape(static_cast<unsigned int>(newShape.size()), newShape.data());
}

TensorInfo ReduceDims(const TensorInfo& tensorInfo, unsigned int dimensions)
{
    TensorInfo strippedTensor(tensorInfo);
    TensorShape strippedShape = ReduceDims(tensorInfo.GetShape(), dimensions);
    strippedTensor.SetShape(strippedShape);
    return strippedTensor;
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

    return { outputDim, outputShape.data() };
}

TensorShape ExpandDimsToRank(const TensorShape& tensorShape, unsigned int rank)
{
    // Can't expand if rank is smaller than current shape
    if (tensorShape.GetNumDimensions() >= rank)
    {
        return tensorShape;
    }

    std::vector<unsigned int> newShape;

    // First add 1s to the beginning of the tensorInfo to fill in the space
    for (unsigned int i = 0; i < rank - tensorShape.GetNumDimensions(); ++i)
    {
        newShape.push_back(1);
    }

    // Then iterate through the original shape and append it to the new shape with the added 1s
    for (unsigned int i = 0; i < tensorShape.GetNumDimensions(); ++i)
    {
        newShape.push_back(tensorShape[i]);
    }

    return TensorShape(static_cast<unsigned int>(newShape.size()), newShape.data());
}

std::vector<unsigned int> SqueezeDims(const TensorShape& tensorShape)
{
    std::vector<unsigned int> squeezedDims;

    for (unsigned int i = 0; i < tensorShape.GetNumDimensions(); ++i)
    {
        if (tensorShape[i] != 1)
        {
            squeezedDims.push_back(tensorShape[i]);
        }
    }
    return squeezedDims;
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

template<typename PrimitiveType>
void CheckSizes(const std::vector<PrimitiveType>& data, const armnn::TensorInfo& tensorInfo, unsigned int size = 1)
{
    if (data.size() / size != tensorInfo.GetNumElements())
    {
        throw InvalidArgumentException(
                fmt::format("The data does not contain the expected number of elements {} != {}. {}",
                            data.size(), tensorInfo.GetNumElements(), CHECK_LOCATION().AsString()));
    }
}

template<typename PrimitiveType>
std::unique_ptr<float[]> ToFloatArray(const std::vector<PrimitiveType>& data, const armnn::TensorInfo& tensorInfo)
{
    CheckSizes(data, tensorInfo);

    std::unique_ptr<float[]> returnBuffer(new float[tensorInfo.GetNumElements()]);

    if (tensorInfo.HasPerAxisQuantization())
    {
        unsigned int axis = tensorInfo.GetQuantizationDim().value();
        auto axisDimensionality = tensorInfo.GetShape()[axis];
        auto axisFactor = armnnUtils::GetNumElementsAfter(tensorInfo.GetShape(), axis);

        for (unsigned int i = 0; i < tensorInfo.GetNumElements(); ++i)
        {
            unsigned int axisIndex;

            if (i < axisFactor)
            {
                axisIndex = 0;
            }
            else
            {
                axisIndex = (i / axisFactor) % axisDimensionality;
            }
            returnBuffer[i] = Dequantize<PrimitiveType>(data[i],
                                                        tensorInfo.GetQuantizationScales()[axisIndex],
                                                        tensorInfo.GetQuantizationOffset());
        }
    }
    else
    {
        for (unsigned int i = 0; i < tensorInfo.GetNumElements(); ++i)
        {
            returnBuffer[i] = Dequantize<PrimitiveType>(data[i],
                                                        tensorInfo.GetQuantizationScale(),
                                                        tensorInfo.GetQuantizationOffset());
        }
    }
    return returnBuffer;
}

std::unique_ptr<float[]> ToFloatArray(const std::vector<uint8_t>& data, const armnn::TensorInfo& tensorInfo)
{
    if (tensorInfo.GetDataType() == DataType::QAsymmS8 || tensorInfo.GetDataType() == DataType::QSymmS8)
    {
        CheckSizes(data, tensorInfo);
        std::vector<int8_t> buffer(tensorInfo.GetNumElements());
        ::memcpy(buffer.data(), data.data(), data.size());
        return ToFloatArray<int8_t>(buffer, tensorInfo);
    }
    else if (tensorInfo.GetDataType() == DataType::QAsymmU8)
    {
        CheckSizes(data, tensorInfo);
        return ToFloatArray<uint8_t>(data, tensorInfo);
    }
    else if (tensorInfo.GetDataType() == DataType::Signed32)
    {
        CheckSizes(data, tensorInfo, 4);
        std::vector<int32_t> buffer(tensorInfo.GetNumElements());
        ::memcpy(buffer.data(), data.data(), data.size());
        return ToFloatArray<int32_t>(buffer, tensorInfo);
    }
    else if (tensorInfo.GetDataType() == DataType::Signed64)
    {
        CheckSizes(data, tensorInfo, 8);
        std::vector<int64_t> buffer(tensorInfo.GetNumElements());
        ::memcpy(buffer.data(), data.data(), data.size());
        return ToFloatArray<int64_t>(buffer, tensorInfo);
    }
    throw InvalidArgumentException(
            fmt::format("Unsupported datatype {}. {}",
                        GetDataTypeName(tensorInfo.GetDataType()),
                        CHECK_LOCATION().AsString()));
}

} // namespace armnnUtils
