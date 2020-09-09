//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "StridedSlice.hpp"

#include <ResolveType.hpp>

#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <cstring>

namespace armnn
{

namespace
{

void PadParams(StridedSliceDescriptor& p, unsigned int dimCount)
{
    ARMNN_ASSERT_MSG(dimCount <= 4, "Expected input with at most 4 dimensions");

    const unsigned int beginIndicesCount =
        armnn::numeric_cast<unsigned int>(p.m_Begin.size());

    ARMNN_ASSERT(dimCount >= beginIndicesCount);
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
    const unsigned char* input = reinterpret_cast<const unsigned char*>(inputData);
    unsigned char* output = reinterpret_cast<unsigned char*>(outputData);

    const TensorShape inputShape = ExtendShape(inputInfo.GetShape(), 4);

    StridedSliceDescriptor paddedParams = params;

    // Pad parameters to 4 dimensions
    PadParams(paddedParams, 4);

    const int start0 = paddedParams.GetStartForAxis(inputShape, 0);
    const int stop0  = paddedParams.GetStopForAxis (inputShape, 0, start0);

    const int start1 = paddedParams.GetStartForAxis(inputShape, 1);
    const int stop1  = paddedParams.GetStopForAxis (inputShape, 1, start1);

    const int start2 = paddedParams.GetStartForAxis(inputShape, 2);
    const int stop2  = paddedParams.GetStopForAxis (inputShape, 2, start2);

    const int start3 = paddedParams.GetStartForAxis(inputShape, 3);
    const int stop3  = paddedParams.GetStopForAxis (inputShape, 3, start3);

    const int step = armnn::numeric_cast<int>(dataTypeSize);

    for (int in0 = start0;
         !LoopCondition(in0, stop0, paddedParams.m_Stride[0]);
         in0 += paddedParams.m_Stride[0])
    {
        for (int in1 = start1;
             !LoopCondition(in1, stop1, paddedParams.m_Stride[1]);
             in1 += paddedParams.m_Stride[1])
        {
            for (int in2 = start2;
                 !LoopCondition(in2, stop2, paddedParams.m_Stride[2]);
                 in2 += paddedParams.m_Stride[2])
            {
                for (int in3 = start3;
                     !LoopCondition(in3, stop3, paddedParams.m_Stride[3]);
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
