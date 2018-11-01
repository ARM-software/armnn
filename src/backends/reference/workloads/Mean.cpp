//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Mean.hpp"
#include "backendsCommon/WorkloadData.hpp"

#include <boost/numeric/conversion/cast.hpp>

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>

namespace armnn
{
bool NextIndex(const unsigned int numDims, const armnn::TensorShape& dims, std::vector<unsigned int>& current)
{
    unsigned int carry = 1;

    for (unsigned int idx = numDims; idx-- > 0; )
    {
        unsigned int current_val = current[idx] + carry;
        if (dims[idx] == current_val)
        {
            current[idx] = 0;
        }
        else
        {
            current[idx] = current_val;
            carry = 0;
            break;
        }
    }
    return (carry == 0);
}

std::size_t ReducedOutputOffset(const unsigned int numDims, const armnn::TensorShape& dims,
                                std::vector<unsigned int>& index, const unsigned int numAxis,
                                const std::vector<unsigned int>& axis) {
    std::size_t offset = 0;
    for (unsigned int idx = 0; idx < numDims; ++idx)
    {
        bool isAxis = false;
        if (!axis.empty())
        {
            for (unsigned int axisIdx = 0; axisIdx < numAxis; ++axisIdx)
            {
                if (idx == axis[axisIdx])
                {
                    isAxis = true;
                    break;
                }
            }
        }
        if (!isAxis)
        {
            offset = offset * boost::numeric_cast<size_t>(dims[idx]) + boost::numeric_cast<size_t>(index[idx]);
        }
    }
    return offset;
}
} // namespace

namespace armnn
{
void Mean(const armnn::TensorInfo& inputInfo,
          const armnn::TensorInfo& outputInfo,
          const std::vector<unsigned int>& axis,
          const float* inputData,
          float* outputData) {

    unsigned int inputNumDims = inputInfo.GetNumDimensions();
    unsigned int outputNumDims = outputInfo.GetNumDimensions();

    armnn::TensorShape outputDims = outputInfo.GetShape();
    armnn::TensorShape inputDims = inputInfo.GetShape();

    // Initialise output data.
    size_t numOutputs = 1;
    for (unsigned int idx = 0; idx < outputNumDims; ++idx)
    {
        numOutputs *= boost::numeric_cast<size_t>(outputDims[idx]);
    }

    std::vector<float> tempSum(numOutputs);
    for (size_t idx = 0; idx < numOutputs; ++idx)
    {
        outputData[idx] = 0.0f;
        tempSum[idx] = 0.0f;
    }

    // Initialise temp index.
    std::vector<unsigned int> tempIndex(inputNumDims);
    for (unsigned int idx = 0; idx < inputNumDims; ++idx)
    {
        tempIndex[idx] = 0;
    }

    std::vector<unsigned int> resolvedAxis = axis;
    if (resolvedAxis.empty())
    {
      for (unsigned int idx = 0; idx < inputNumDims; ++idx)
      {
          resolvedAxis.push_back(idx);
      }
    }
    unsigned int numResolvedAxis = boost::numeric_cast<unsigned int>(resolvedAxis.size());

    // Iterates through input_data and sum up the reduced axis.
    for (bool hasNext = true; hasNext; hasNext = NextIndex(inputNumDims, inputDims, tempIndex))
    {
        size_t inputOffset = ReducedOutputOffset(inputNumDims, inputDims, tempIndex, 0, {});
        size_t outputOffset = ReducedOutputOffset(inputNumDims, inputDims, tempIndex,
                                                  numResolvedAxis, resolvedAxis);
        tempSum[outputOffset] += inputData[inputOffset];
    }

    // Takes average by num of elements added to get mean.
    size_t numElementsInAxis = 1;
    for (unsigned int idx = 0; idx < numResolvedAxis; ++idx)
    {
        size_t current = boost::numeric_cast<size_t>(inputDims[resolvedAxis[idx]]);
        BOOST_ASSERT(boost::numeric_cast<float>(current) <
              (std::numeric_limits<float>::max() / boost::numeric_cast<float>(numElementsInAxis)));
        numElementsInAxis *= current;
    }
    if (numElementsInAxis > 0) {
        for (size_t idx = 0; idx < numOutputs; ++idx)
        {
            outputData[idx] = tempSum[idx] / boost::numeric_cast<float>(numElementsInAxis);
        }
    }
}
} //namespace armnn
