//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserHelper.hpp"

#include <armnn/Descriptors.hpp>
#include <armnnUtils/Permute.hpp>

#include <fmt/format.h>

namespace armnnUtils
{

const armnn::PermutationVector NHWCToArmNN = { 0, 2, 3, 1 };
const armnn::PermutationVector ArmNNToNHWC = { 0, 3, 1, 2 };

void ProcessConcatInputTensorInfo(armnn::TensorInfo& inputTensorInfo,
                                  armnn::OriginsDescriptor& concatDescriptor,
                                  const unsigned int& concatAxis,
                                  unsigned int inputIndex,
                                  unsigned int& mergeDimOrigin)
{
    const uint32_t inputRank = concatDescriptor.GetNumDimensions();

    // double check dimensions of the tensors
    if (inputTensorInfo.GetNumDimensions() != inputRank)
    {
        throw armnn::ParseException(fmt::format(
                                    "The number of dimensions: {0} for input tensors of the "
                                    "concatenation op should be {1} {2}",
                                    inputTensorInfo.GetNumDimensions(),
                                    inputRank,
                                    CHECK_LOCATION().AsString()));
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

void CalculateReducedOutputTensoInfo(const armnn::TensorInfo& inputTensorInfo,
                                     const std::set<unsigned int>& axisSet,
                                     bool keepDims,
                                     armnn::TensorInfo& outputTensorInfo)
{
    std::vector<unsigned int> outputShapeVector;
    bool dimensionFound = false;
    unsigned int size = 1;

    for (unsigned int i = 0; i < inputTensorInfo.GetNumDimensions(); ++i)
    {
        dimensionFound = false;
        for (unsigned int axis: axisSet)
        {
            if (axis == i)
            {
                dimensionFound = true;
                break;
            }
        }

        if (!dimensionFound)
        {
            size *= inputTensorInfo.GetShape()[i];

            if (keepDims)
            {
                outputShapeVector.push_back(inputTensorInfo.GetShape()[i]);
            }
        }
        else
        {
            if (keepDims)
            {
                outputShapeVector.push_back(1);
            }
        }
    }

    if (keepDims)
    {
        armnn::TensorShape outputTensorShape(inputTensorInfo.GetNumDimensions(), &outputShapeVector[0]);
        outputTensorInfo = armnn::TensorInfo(outputTensorShape, inputTensorInfo.GetDataType());
    }
    else
    {
        outputTensorInfo = armnn::TensorInfo({size}, inputTensorInfo.GetDataType());
    }
}


void CalculateStridedSliceOutputTensorInfo(const armnn::TensorInfo& inputTensorInfo,
                                           const armnn::StridedSliceDescriptor& desc,
                                           armnn::TensorInfo& outputTensorInfo)
{
    const armnn::TensorShape& inputShape = inputTensorInfo.GetShape();

    std::vector<unsigned int> outputShapeVector;
    for (unsigned int i = 0; i < inputTensorInfo.GetNumDimensions(); i++)
    {
        if (desc.m_ShrinkAxisMask & (1 << i))
        {
            continue;
        }

        int stride = desc.m_Stride[i];
        int start = desc.GetStartForAxis(inputShape, i);
        int stop = desc.GetStopForAxis(inputShape, i, start);

        int newSize = stride > 0 ? ((stop - start) + stride - 1) / stride :
                      ((start - stop) - stride - 1) / -stride;

        newSize = std::max(0, newSize);

        outputShapeVector.push_back(static_cast<unsigned int>(newSize));
    }

    armnn::TensorShape outputTensorShape(inputTensorInfo.GetNumDimensions(), &outputShapeVector[0]);
    outputTensorInfo = armnn::TensorInfo(armnn::TensorShape(outputTensorShape), inputTensorInfo.GetDataType());
}
} // namespace armnnUtils
