//
// Copyright © 2020-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn_delegate.hpp>

#include <armnn/ArmNN.hpp>
#include <armnn/BackendHelper.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <armnnUtils/Permute.hpp>
#include <armnnUtils/TensorUtils.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <tensorflow/lite/kernels/kernel_util.h>

#include <numeric>

namespace
{

armnn::DataType GetDataType(const TfLiteTensor& tfLiteTensor)
{
    switch (tfLiteTensor.type)
    {
        case kTfLiteBool:
            return armnn::DataType::Boolean;
        case kTfLiteFloat32:
            return armnn::DataType::Float32;
        case kTfLiteFloat16:
            return armnn::DataType::Float16;
        case kTfLiteUInt8:
            return armnn::DataType::QAsymmU8;
        case kTfLiteInt8:
        {
            auto quantizationInfo = tfLiteTensor.quantization;
            if (quantizationInfo.type == kTfLiteAffineQuantization)
            {
                auto* quantization =
                    reinterpret_cast<TfLiteAffineQuantization*>(tfLiteTensor.quantization.params);
                if (quantization->zero_point != nullptr && quantization->zero_point->size == 1)
                {
                    return armnn::DataType::QAsymmS8;
                }
                else
                {
                    return armnn::DataType::QSymmS8;
                }
            }
            else
            {
                return armnn::DataType::QAsymmS8;
            }
        }
        case kTfLiteInt16:
            return armnn::DataType::QSymmS16;
        case kTfLiteInt32:
            return armnn::DataType::Signed32;
        case kTfLiteInt64:
            return armnn::DataType::Signed64;
        default:
            throw armnn::Exception(&"TfLiteArmnnDelegate: Unsupported data type: " [ tfLiteTensor.type]);
    }
}

uint32_t NonNegative(int32_t value, int nodeIndex)
{
    if (value < 0)
    {
        throw armnn::Exception(
                "TfLiteArmnnDelegate: Non-negative value in node " + std::to_string(static_cast<int>(nodeIndex)));
    }
    else
    {
        return static_cast<uint32_t>(value);
    }
}

void ExpandTensorRankToEqual(armnn::TensorInfo& inputInfo0,
                             armnn::TensorInfo& inputInfo1)
{
    unsigned int inputDimensions0 = inputInfo0.GetNumDimensions();
    unsigned int inputDimensions1 = inputInfo1.GetNumDimensions();

    if (inputDimensions0 == inputDimensions1)
    {
        return;
    }

    unsigned int biggerInputDimensions = std::max(inputDimensions0, inputDimensions1);

    bool input0IsSmaller = inputDimensions0 < inputDimensions1;
    armnn::TensorInfo& smallInfo = input0IsSmaller ? inputInfo0 : inputInfo1;
    const armnn::TensorShape& newShape = armnnUtils::ExpandDimsToRank(smallInfo.GetShape(), biggerInputDimensions);

    smallInfo.SetShape(newShape);
}

void CalcPadding(uint32_t inputSize,
                 uint32_t filterSize,
                 uint32_t stride,
                 uint32_t dilation,
                 uint32_t& paddingFront,
                 uint32_t& paddingBack,
                 TfLitePadding padding)
{
    paddingFront = 0;
    paddingBack = 0;
    if (padding == kTfLitePaddingSame)
    {
        uint32_t outputSize = (inputSize + stride - 1) / stride;
        uint32_t dilatedSize = filterSize + (dilation - 1) * (filterSize - 1);
        uint32_t temp = (outputSize - 1) * stride + dilatedSize;
        if (temp > inputSize)
        {
            paddingFront = (temp - inputSize) / 2;
            paddingBack = (temp - inputSize) - paddingFront;
        }
    }
}

// Function that calculates explicit padding when the output shape is known.
// At the moment the output is only given as an input parameter in Transpose Convolution,
// not in Convolution and Depthwise Convolution
void CalcPadding(uint32_t inputSize,
                 uint32_t filterSize,
                 uint32_t stride,
                 uint32_t dilation,
                 uint32_t& paddingFront,
                 uint32_t& paddingBack,
                 TfLitePadding padding,
                 uint32_t outputSize)
{
    armnn::IgnoreUnused(dilation);
    paddingFront = 0;
    paddingBack = 0;
    if (padding == kTfLitePaddingSame)
    {
        uint32_t totalPadding = (inputSize - 1) * stride + filterSize - outputSize;
        paddingFront = totalPadding / 2;
        paddingBack = totalPadding - paddingFront;
    }
}

unsigned int ComputeWrappedIndex(int index, unsigned int numDimensions)
{
    int numDims = armnn::numeric_cast<int>(numDimensions);
    int wrappedIndex = index < 0 ? numDims + index : index;

    if (wrappedIndex < 0 || wrappedIndex >= numDims)
    {
        throw armnn::ParseException("Unable to compute wrapped index");
    }

    return static_cast<unsigned int>(wrappedIndex);
};

bool AreAllSigned32(const armnn::TensorInfo& inputInfo1,
                    const armnn::TensorInfo& inputInfo2,
                    const armnn::TensorInfo& outputInfo)
{
    return (armnn::DataType::Signed32 == inputInfo1.GetDataType()) &&
           (armnn::DataType::Signed32 == inputInfo2.GetDataType()) &&
           (armnn::DataType::Signed32 == outputInfo.GetDataType());
}

void UpdateConstantTensorOutputs(const armnn::TensorInfo& inputInfo, armnn::TensorInfo& outputInfo)
{
    // If input tensor info is constant and output tensor info shape is not specified
    // set the output shape from input shape
    if (inputInfo.IsConstant() && outputInfo.GetShape().GetDimensionality() == armnn::Dimensionality::NotSpecified)
    {
        outputInfo.SetShape(inputInfo.GetShape());
    }
}

void SetupConcatViewOrigin(const armnn::TensorInfo& inputTensorInfo,
                           armnn::OriginsDescriptor& concatDescriptor,
                           const unsigned int concatAxis,
                           unsigned int inputIndex,
                           unsigned int& mergeDimOrigin)
{
    const uint32_t inputRank = concatDescriptor.GetNumDimensions();

    // double check dimensions of the tensors
    if (inputTensorInfo.GetNumDimensions() != inputRank)
    {
        throw armnn::ParseException("The number of dimensions for input tensors "
                                    "of the concatenation operator should be: " + std::to_string(inputRank));
    }

    for (unsigned int j = 0; j < concatAxis; ++j)
    {
        concatDescriptor.SetViewOriginCoord(inputIndex, j, 0);
    }

    concatDescriptor.SetViewOriginCoord(inputIndex, concatAxis, mergeDimOrigin);
    mergeDimOrigin += inputTensorInfo.GetShape()[concatAxis];

    for (unsigned int j = concatAxis + 1; j < inputRank; ++j)
    {
        concatDescriptor.SetViewOriginCoord(inputIndex, j, 0);
    }
}

TfLiteStatus CreateOutputTensorShape(const armnn::TensorInfo& inputTensorInfo,
                                     const std::vector<int32_t>& targetShape,
                                     armnn::ReshapeDescriptor& reshapeDesc)
{
    std::vector<unsigned int> outputDims(targetShape.begin(), targetShape.end());
    const auto stretchDim = std::find(targetShape.begin(), targetShape.end(), -1);

    if (stretchDim != targetShape.end())
    {
        if (std::find(std::next(stretchDim), targetShape.end(), -1) != targetShape.end())
        {
            // Return kTfLiteError and log the error after returning
            return kTfLiteError;
        }

        auto targetNumElements =
                armnn::numeric_cast<unsigned int>(
                        std::accumulate(targetShape.begin(), targetShape.end(), -1, std::multiplies<int32_t>()));

        auto stretchIndex = static_cast<size_t>(std::distance(targetShape.begin(), stretchDim));

        if (targetNumElements == 0)
        {
            // To handle the edge case that input and output both have zero elements
            outputDims[stretchIndex] = 0;
        }
        else
        {
            outputDims[stretchIndex] = inputTensorInfo.GetNumElements() / targetNumElements;
        }
    }

    armnn::TensorShape outputShape = armnn::TensorShape(static_cast<unsigned int>(outputDims.size()),
                                                        outputDims.data());
    reshapeDesc.m_TargetShape = outputShape;
    return kTfLiteOk;
}

armnn::TensorInfo OutputShapeOfSqueeze(std::vector<uint32_t> squeezeDims,
                                       const armnn::TensorInfo& inputTensorInfo)
{
    static const uint32_t dimensionSequence[] = { 0, 1, 2, 3 };

    if (inputTensorInfo.GetNumDimensions() > 4)
    {
        std::stringstream ss;
        ss << "Input tensor has unexpected number of dimensions:"
           << inputTensorInfo.GetNumDimensions()
           << " shape:" << inputTensorInfo.GetShape()
           << " "
           << CHECK_LOCATION().AsString();
        throw armnn::ParseException(ss.str());
    }

    if (squeezeDims.empty())
    {
        squeezeDims.assign(dimensionSequence, dimensionSequence + inputTensorInfo.GetNumDimensions());
    }

    std::vector<uint32_t> outputDims;
    for(unsigned int i = 0; i < inputTensorInfo.GetNumDimensions(); i++)
    {
        bool skipSqueeze = (std::find(squeezeDims.begin(), squeezeDims.end(), i) == squeezeDims.end());
        auto currentDimension = inputTensorInfo.GetShape()[i];
        if (skipSqueeze || currentDimension != 1)
        {
            outputDims.push_back(currentDimension);
        }
    }

    if (outputDims.size() > 4)
    {
        std::stringstream ss;
        ss << "Output tensor has unexpected number of dimensions:"
           << inputTensorInfo.GetNumDimensions()
           << " shape:" << inputTensorInfo.GetShape()
           << " "
           << CHECK_LOCATION().AsString();
        throw armnn::ParseException(ss.str());
    }

    armnn::TensorShape outShape = armnn::TensorShape(static_cast<unsigned int>(outputDims.size()), outputDims.data());

    // We need to preserve the tensor type and the quantization data as well
    armnn::TensorInfo outTensorInfo = inputTensorInfo;
    outTensorInfo.SetShape(outShape);

    return outTensorInfo;
}

bool ZeroDimPresent(std::initializer_list<armnn::TensorInfo> tensorInfoList)
{
    for (armnn::TensorInfo tensorInfo : tensorInfoList)
    {
        for (unsigned int i = 0; i < tensorInfo.GetNumDimensions(); ++i)
        {
            if (tensorInfo.GetShape()[i] == 0)
            {
                return true;
            }
            if (tensorInfo.IsQuantized() && tensorInfo.GetQuantizationDim() == 0)
            {
                return true;
            }
        }
    }
    return false;
}

} // namespace anonymous
