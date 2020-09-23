//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Slice.hpp"

#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

namespace armnn
{

void Slice(const TensorInfo& inputInfo,
           const SliceDescriptor& descriptor,
           const void* inputData,
           void* outputData,
           unsigned int dataTypeSize)
{
    const TensorShape& inputShape = inputInfo.GetShape();
    const unsigned int numDims    = inputShape.GetNumDimensions();

    ARMNN_ASSERT(descriptor.m_Begin.size() == numDims);
    ARMNN_ASSERT(descriptor.m_Size.size()  == numDims);

    constexpr unsigned int maxNumDims = 4;
    ARMNN_ASSERT(numDims <= maxNumDims);

    std::vector<unsigned int> paddedInput(4);
    std::vector<unsigned int> paddedBegin(4);
    std::vector<unsigned int> paddedSize (4);

    const unsigned int numPaddingDims = maxNumDims - numDims;
    for (unsigned int i = 0u; i < maxNumDims; ++i)
    {
        if (i < numPaddingDims)
        {
            paddedInput[i] = 1u;
            paddedBegin[i] = 0u;
            paddedSize[i]  = 1u;
        }
        else
        {
            const unsigned int j = i - numPaddingDims;
            paddedInput[i] = inputShape[j];
            paddedBegin[i] = descriptor.m_Begin[j];
            paddedSize[i]  = descriptor.m_Size[j];
        }
    }

    unsigned int dim0 = paddedInput[0];
    unsigned int dim1 = paddedInput[1];
    unsigned int dim2 = paddedInput[2];
    unsigned int dim3 = paddedInput[3];

    unsigned int begin0 = paddedBegin[0];
    unsigned int begin1 = paddedBegin[1];
    unsigned int begin2 = paddedBegin[2];
    unsigned int begin3 = paddedBegin[3];

    unsigned int size0  = paddedSize[0];
    unsigned int size1  = paddedSize[1];
    unsigned int size2  = paddedSize[2];
    unsigned int size3  = paddedSize[3];

    ARMNN_ASSERT(begin0 + size0 <= dim0);
    ARMNN_ASSERT(begin1 + size1 <= dim1);
    ARMNN_ASSERT(begin2 + size2 <= dim2);
    ARMNN_ASSERT(begin3 + size3 <= dim3);

    const unsigned char* input = reinterpret_cast<const unsigned char*>(inputData);
    unsigned char* output      = reinterpret_cast<unsigned char*>(outputData);

    IgnoreUnused(dim0);
    for (unsigned int idx0 = begin0; idx0 < begin0 + size0; ++idx0)
    {
        for (unsigned int idx1 = begin1; idx1 < begin1 + size1; ++idx1)
        {
            for (unsigned int idx2 = begin2; idx2 < begin2 + size2; ++idx2)
            {
                for (unsigned int idx3 = begin3; idx3 < begin3 + size3; ++idx3)
                {
                    const unsigned int inputOffset =
                        (((idx0 * dim1 + idx1) * dim2 + idx2) * dim3 + idx3) * dataTypeSize;

                    ::memcpy(output, input + inputOffset, dataTypeSize);
                    output += dataTypeSize;
                }
            }
        }
    }
}

} // namespace armnn
