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

    constexpr unsigned int maxNumDims = 4;
    if (descriptor.m_Begin.size() != numDims)
    {
        std::stringstream msg;
        msg << "Slice: Number of dimensions (" << numDims <<
            ") does not match the Begin vector in the descriptor (" << descriptor.m_Begin.size() << ")";
        throw InvalidArgumentException(msg.str());
    }
    if (descriptor.m_Size.size() != numDims)
    {
        std::stringstream msg;
        msg << "Slice: Number of dimensions (" << numDims <<
            ") does not match the Size vector in the descriptor (" << descriptor.m_Size.size() << ")";
        throw InvalidArgumentException(msg.str());
    }
    if (numDims > maxNumDims)
    {
        std::stringstream msg;
        msg << "Slice: Number of dimensions (" << numDims <<
            ") is greater than the maximum supported (" << maxNumDims << ")";
        throw InvalidArgumentException(msg.str());
    }

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

    if (begin0 + size0 > dim0)
    {
        std::stringstream msg;
        msg << "Slice: begin0 + size0 (" << (begin0 + size0) <<
            ") exceeds dim0 (" << dim0 << ")";
        throw InvalidArgumentException(msg.str());
    }
    if (begin1 + size1 > dim1)
    {
        std::stringstream msg;
        msg << "Slice: begin1 + size1 (" << (begin1 + size1) <<
            ") exceeds dim2 (" << dim1 << ")";
        throw InvalidArgumentException(msg.str());
    }
    if (begin2 + size2 > dim2)
    {
        std::stringstream msg;
        msg << "Slice: begin2 + size2 (" << (begin2 + size2) <<
            ") exceeds dim2 (" << dim2 << ")";
        throw InvalidArgumentException(msg.str());
    }
    if (begin3 + size3 > dim3)
    {
        std::stringstream msg;
        msg << "Slice: begin3 + size3 (" << (begin3 + size3) <<
            ") exceeds dim3 (" << dim3 << ")";
        throw InvalidArgumentException(msg.str());
    }

    if (inputData == nullptr)
    {
        throw armnn::NullPointerException("Slice: Null inputData pointer");
    }
    if (outputData == nullptr)
    {
        throw armnn::NullPointerException("Slice: Null outputData pointer");
    }

    const unsigned char* input = reinterpret_cast<const unsigned char*>(inputData);
    unsigned char* output      = reinterpret_cast<unsigned char*>(outputData);

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
