//
// Copyright © 2017, 2024 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "StridedSlice.hpp"

#include <armnn/utility/NumericCast.hpp>

#include <cstring>

namespace armnn
{

namespace
{

void PadParams(StridedSliceDescriptor& p, unsigned int dimCount)
{
    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(dimCount <= 4, "Expected input with at most 4 dimensions");

    const unsigned int beginIndicesCount =
        armnn::numeric_cast<unsigned int>(p.m_Begin.size());

    const unsigned int padCount = dimCount - beginIndicesCount;

    p.m_Begin.resize(dimCount);
    p.m_End.resize(dimCount);
    p.m_Stride.resize(dimCount);

    for (unsigned int i = beginIndicesCount; i > 0; --i)
    {
        p.m_Stride[i + padCount - 1] = p.m_Stride[i - 1];
        p.m_Begin[i + padCount - 1] = p.m_Begin[i - 1];
        p.m_End[i + padCount - 1] = p.m_End[i - 1];
    }

    for (unsigned int i = 0; i < padCount; ++i)
    {
        p.m_Stride[i] = 1;
        p.m_Begin[i] = 0;
        p.m_End[i] = 0;
    }

    p.m_ShrinkAxisMask <<= padCount;
    p.m_EllipsisMask <<= padCount;
    p.m_NewAxisMask <<= padCount;
    p.m_BeginMask <<= padCount;
    p.m_EndMask <<= padCount;
    p.m_BeginMask |= (1 << padCount) - 1;
    p.m_EndMask |= (1 << padCount) - 1;
}

bool LoopCondition(int index, int stop, int stride)
{
    return stride > 0 ? index >= stop : index <= stop;
}

TensorShape ExtendShape(const TensorShape& inputShape,
                        unsigned int newNumDimensions)
{
    if (inputShape.GetNumDimensions() >= newNumDimensions)
    {
        return inputShape;
    }

    std::vector<unsigned int> newSizes(newNumDimensions, 0);

    unsigned int diff = newNumDimensions - inputShape.GetNumDimensions();

    for (unsigned int i = 0; i < diff; i++)
    {
        newSizes[i] = 1;
    }

    for (unsigned int i = diff; i < newNumDimensions; i++)
    {
        newSizes[i] = inputShape[i - diff];
    }

    return TensorShape(newNumDimensions, newSizes.data());
}

} // Anonymous namespace

void StridedSlice(const TensorInfo& inputInfo,
                  const StridedSliceDescriptor& params,
                  const void* inputData,
                  void* outputData,
                  unsigned int dataTypeSize)
{
    if (inputData == nullptr)
    {
        throw armnn::InvalidArgumentException("Slice: Null inputData pointer");
    }
    if (outputData == nullptr)
    {
        throw armnn::InvalidArgumentException("Slice: Null outputData pointer");
    }

    const unsigned char* input = reinterpret_cast<const unsigned char*>(inputData);
    unsigned char* output = reinterpret_cast<unsigned char*>(outputData);

    const TensorShape inputShape = ExtendShape(inputInfo.GetShape(), 4);

    StridedSliceDescriptor paddedParams = params;

    // Pad parameters to 4 dimensions
    PadParams(paddedParams, 4);

    // Arrays containing the start and stop index for each axis (adjusted by set params/flags)
    int startArray [4] = {0};
    int stopArray [4] = {0};

    // Getting paddedParams stop and start values for each axis
    for(unsigned int i = 0; i < 4; ++i)
    {
        startArray[i] = paddedParams.GetStartForAxis(inputShape, i);
        stopArray[i] = paddedParams.GetStopForAxis(inputShape, i, startArray[i]);
    }

    // Adjusting the EllipsisMask based on the NewAxisMask
    // (if NewAxisMask extends an axis, the ellipsis flag is extended as well)
    if(paddedParams.m_NewAxisMask > 0 && paddedParams.m_EllipsisMask > 0)
    {
        // Iterate until the current EllipsisMask 1-bit found
        for(unsigned int i = 0; i < 4; ++i)
        {
            // If EllipsisMask bit found, adjust based on NewAxisMask and exit loop
            if(paddedParams.m_EllipsisMask & (1 << i) && !(paddedParams.m_NewAxisMask & (1 << i)))
            {
                // If the previous bit is the NewAxisMask, set the EllipsisMask there
                // (this condition was determined based on the unit tests expected data)
                if(paddedParams.m_NewAxisMask & (1 << (i-1)))
                {
                    paddedParams.m_EllipsisMask |= (1 << (i-1));
                }
                // Otherwise, extend the EllipsisMask by one bit
                else
                {
                    paddedParams.m_EllipsisMask |= (1 << (i+1));
                }
                break;
            }
        }
    }

    // Processing start and stop values based on the EllipsisMask and NewAxisMask
    for(unsigned int i = 0, dimIdx = 0; i < 4; ++i)
    {
        // If the EllipsisMask is set, extend the start/stop to the input dimension size
        if(paddedParams.m_EllipsisMask & (1 << dimIdx))
        {
            startArray[i] = 0;
            stopArray[i] = armnn::numeric_cast<int>(inputShape[i]);
        }
        // Otherwise, if the NewAxisMask is set, shift all following start/stop values to the left
        else if(paddedParams.m_NewAxisMask & (1 << dimIdx))
        {
            // Increment dimIdx - skip the current dimension for which NewAxisMask is set
            ++dimIdx;
        }

        // If the index of the currently processed dimension is higher than
        // the index of the current start/stop array position, shift start/stop values
        if(dimIdx > i && !(paddedParams.m_EllipsisMask & (1 << dimIdx)))
        {
            if(dimIdx < 4)
            {
                startArray[i] = startArray[dimIdx];
                stopArray[i] = stopArray[dimIdx];
            }
            else
            {
                // If dimIdx is greater than the amount of available dimensions,
                // instead of shifting the next ones, create new start/stop values
                if(paddedParams.m_EllipsisMask > 0)
                {
                    // The new values are 0,1 if there is an EllipsisMask bit present
                    startArray[i] = 0;
                    stopArray[i] = 1;
                }
                else
                {
                    // Otherwise, select the entire inputTensor dimension size
                    startArray[i] = 0;
                    stopArray[i] = armnn::numeric_cast<int>(inputShape[i]);
                }
            }
        }
        ++dimIdx;
    }

    const int step = armnn::numeric_cast<int>(dataTypeSize);

    for (int in0 = startArray[0];
         !LoopCondition(in0, stopArray[0], paddedParams.m_Stride[0]);
         in0 += paddedParams.m_Stride[0])
    {
        for (int in1 = startArray[1];
             !LoopCondition(in1, stopArray[1], paddedParams.m_Stride[1]);
             in1 += paddedParams.m_Stride[1])
        {
            for (int in2 = startArray[2];
                 !LoopCondition(in2, stopArray[2], paddedParams.m_Stride[2]);
                 in2 += paddedParams.m_Stride[2])
            {
                for (int in3 = startArray[3];
                     !LoopCondition(in3, stopArray[3], paddedParams.m_Stride[3]);
                     in3 += paddedParams.m_Stride[3])
                {
                    int dim1 = armnn::numeric_cast<int>(inputShape[1]);
                    int dim2 = armnn::numeric_cast<int>(inputShape[2]);
                    int dim3 = armnn::numeric_cast<int>(inputShape[3]);

                    int inputOffset = (((in0 * dim1 + in1) * dim2 + in2) * dim3 + in3) * step;
                    ::memcpy(output, input + inputOffset, dataTypeSize);
                    output += step;
                }
            }
        }
    }
}

} // namespace armnn
