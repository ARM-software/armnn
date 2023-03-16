//
// Copyright © 2020-2023 Arm Ltd and Contributors. All rights reserved.
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

namespace
{

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

unsigned int ComputeWrappedIndex(int index, unsigned int numDimensions)
{
    int numDims = armnn::numeric_cast<int>(numDimensions);
    int wrappedIndex = index < 0 ? numDims + index : index;
    ARMNN_ASSERT(wrappedIndex >= 0);
    ARMNN_ASSERT(wrappedIndex < numDims);

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

} // namespace anonymous
