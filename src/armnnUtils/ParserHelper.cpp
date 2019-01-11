//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserHelper.hpp"

// armnnUtils
#include "Permute.hpp"

#include <boost/format.hpp>

namespace armnnUtils
{

const armnn::PermutationVector NHWCToArmNN = { 0, 2, 3, 1 };
const armnn::PermutationVector ArmNNToNHWC = { 0, 3, 1, 2 };

void ProcessConcatInputTensorInfo(armnn::TensorInfo& inputTensorInfo, armnn::OriginsDescriptor& concatDescriptor,
                                  const unsigned int& concatAxis, unsigned int inputIndex,
                                  std::vector<unsigned int>& mergeDimSizes, unsigned int& mergeDim)
{
    // double check dimensions of the tensors
    if (inputTensorInfo.GetNumDimensions() != armnn::MaxNumOfTensorDimensions)
    {
        throw armnn::ParseException(
            boost::str(
                boost::format(
                    "The number of dimensions: %1% for input tensors of the "
                    "concatenation op should be %2% %3%")
                % inputTensorInfo.GetNumDimensions()
                % armnn::MaxNumOfTensorDimensions
                % CHECK_LOCATION().AsString()));
    }

    // if concatenation axis is 3 then need to be permuted
    if (concatAxis == 3)
    {
        inputTensorInfo = armnnUtils::Permuted(inputTensorInfo, NHWCToArmNN);
    }

    for (unsigned int dim = 0; dim < armnn::MaxNumOfTensorDimensions; ++dim)
    {
        mergeDimSizes[dim] = inputTensorInfo.GetShape()[dim];
    }

    // Concatenation dimension 1 is the only dimension supported in ArmNN
    const unsigned int concatenationDim = 1;

    for (unsigned int j = 0; j < concatenationDim; ++j)
    {
        concatDescriptor.SetViewOriginCoord(inputIndex, j, 0);
    }

    concatDescriptor.SetViewOriginCoord(inputIndex, concatenationDim, mergeDim);
    mergeDim += mergeDimSizes[concatenationDim];

    for (unsigned int j = concatenationDim + 1; j < armnn::MaxNumOfTensorDimensions; ++j)
    {
        concatDescriptor.SetViewOriginCoord(inputIndex, j, 0);
    }
}

void CalculateReducedOutputTensoInfo(const armnn::TensorInfo& inputTensorInfo, const armnn::TensorInfo& axisTensorInfo,
                                     const std::set<unsigned int>& axisSet, bool keepDims,
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

} // namespace armnnUtils
