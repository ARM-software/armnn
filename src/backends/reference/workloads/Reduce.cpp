//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Reduce.hpp"

#include <armnn/utility/NumericCast.hpp>

#include <armnn/backends/WorkloadData.hpp>

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

unsigned int ReducedOutputOffset(const unsigned int numDims,
                                 const armnn::TensorShape& dims,
                                 std::vector<unsigned int>& index,
                                 const unsigned int numAxis,
                                 const std::vector<unsigned int>& axis)
{
    unsigned int offset = 0;
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
            offset = offset * dims[idx] + index[idx];
        }
    }
    return offset;
}


void Reduce(const TensorInfo& inputInfo,
            const TensorInfo& outputInfo,
            Decoder<float>& input,
            Encoder<float>& output,
            const std::vector<uint32_t> axis,
            const ReduceOperation reduceOperation)
{
    armnn::TensorShape inputDims = inputInfo.GetShape();
    unsigned int inputNumDims    = inputInfo.GetNumDimensions();
    unsigned int numOutputs      = outputInfo.GetNumElements();

    // Initialise temp output
    std::vector<float> tempOut(numOutputs);
    switch(reduceOperation)
    {
        case ReduceOperation::Mean:
        case ReduceOperation::Sum:
            std::fill(tempOut.begin(), tempOut.end(), 0.0f);
            break;
        case ReduceOperation::Prod:
            std::fill(tempOut.begin(), tempOut.end(), 1.0f);
            break;
        case ReduceOperation::Max:
            std::fill(tempOut.begin(), tempOut.end(), -1 * std::numeric_limits<float>::max());
            break;
        case ReduceOperation::Min:
            std::fill(tempOut.begin(), tempOut.end(), std::numeric_limits<float>::max());
            break;
        default:
            throw armnn::InvalidArgumentException("Unknown reduce method: " +
                std::to_string(static_cast<int>(reduceOperation)));
    }

    // Initialise temp index
    std::vector<unsigned int> tempIndex(inputNumDims, 0);

    std::vector<unsigned int> resolvedAxis = axis;
    if (resolvedAxis.empty())
    {
        for (unsigned int idx = 0; idx < inputNumDims; ++idx)
        {
            resolvedAxis.push_back(idx);
        }
    }
    auto numResolvedAxis = armnn::numeric_cast<unsigned int>(resolvedAxis.size());

    // Iterates through input_data and operates over the reduced axis
    for (bool hasNext = true; hasNext; hasNext = NextIndex(inputNumDims, inputDims, tempIndex))
    {
        unsigned int inputOffset = ReducedOutputOffset(inputNumDims, inputDims, tempIndex, 0, {});
        unsigned int outputOffset = ReducedOutputOffset(inputNumDims, inputDims, tempIndex,
                                                        numResolvedAxis, resolvedAxis);
        input[inputOffset];
        auto inputValue = input.Get();
        switch(reduceOperation)
        {
            case ReduceOperation::Mean:
            case ReduceOperation::Sum:
                tempOut[outputOffset] += inputValue;
                break;
            case ReduceOperation::Prod:
                tempOut[outputOffset] *= inputValue;
                break;
            case ReduceOperation::Max:
                if (inputValue > tempOut[outputOffset])
                {
                    tempOut[outputOffset] = inputValue;
                }
                break;
            case ReduceOperation::Min:
                if (inputValue < tempOut[outputOffset])
                {
                    tempOut[outputOffset] = inputValue;
                }
                break;
            default:
                throw armnn::InvalidArgumentException("Unknown reduce method: " +
                    std::to_string(static_cast<int>(reduceOperation)));
        }
    }

    // Takes average by num of elements added to get MEAN
    size_t numElementsInAxis = 1;
    for (unsigned int idx = 0; idx < numResolvedAxis; ++idx)
    {
        unsigned int current = inputDims[resolvedAxis[idx]];
        ARMNN_ASSERT(armnn::numeric_cast<float>(current) <
                     (std::numeric_limits<float>::max() / armnn::numeric_cast<float>(numElementsInAxis)));
        numElementsInAxis *= current;
    }

    for (unsigned int idx = 0; idx < numOutputs; ++idx)
    {
        output[idx];
        if (reduceOperation == ReduceOperation::Mean)
        {
            if (numElementsInAxis > 0)
            {
                output.Set(tempOut[idx] / armnn::numeric_cast<float>(numElementsInAxis));
            }
        }
        else
        {
            output.Set(tempOut[idx]);
        }
    }
}

} //namespace armnn