//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Tensor.hpp>

#include <armnnUtils/Transpose.hpp>

#include "Half.hpp"

#include <cassert>
#include <cstring>

namespace
{

class TransposeLoop
{
public:
    using size_type = unsigned int;

    TransposeLoop(const armnn::TensorShape& srcShape, const armnn::PermutationVector& mappings)
        : m_SrcShape(srcShape)
    {
        assert(srcShape.GetNumDimensions() == mappings.GetSize());

        const size_type numDims = srcShape.GetNumDimensions();

        size_type srcStride = 1U;
        size_type dstStride = 1U;

        for (size_type i = numDims - 1U, k = 0U; k < numDims; ++k, --i)
        {
            m_SrcStrides[i] = srcStride;
            m_DstStrides[mappings[i]] = dstStride;

            srcStride *= srcShape[i];
            dstStride *= srcShape[mappings[i]];
        }
    }

    void Unroll(const void* srcData, void* dstData, size_t dataTypeSize)
    {
        assert(srcData);
        assert(dstData);
        assert(dataTypeSize > 0);

        const unsigned char* srcDataPtr = reinterpret_cast<const unsigned char*>(srcData);
        unsigned char* dstDataPtr       = reinterpret_cast<unsigned char*>(dstData);

        const unsigned char* const srcEndPtr = srcDataPtr + m_SrcShape.GetNumElements() * dataTypeSize;
        unsigned char* const       dstEndPtr = dstDataPtr + m_SrcShape.GetNumElements() * dataTypeSize;

        Unroll(0, srcDataPtr, dstDataPtr, srcEndPtr, dstEndPtr, dataTypeSize);
    }

private:
    void Unroll(size_type dimension,
                const unsigned char* srcData, unsigned char* dstData,
                const unsigned char* srcEnd, unsigned char* dstEnd,
                size_t dataTypeSize)
    {
        assert(srcData);
        assert(dstData);
        assert(srcEnd);
        assert(dstEnd);
        assert(srcData < srcEnd);
        assert(dstData < dstEnd);
        assert(dataTypeSize > 0);

        if (dimension >= m_SrcShape.GetNumDimensions())
        {
            ::memcpy(dstData, srcData, dataTypeSize);
        }
        else
        {
            for (size_type i = 0; i < m_SrcShape[dimension]; i++)
            {
                Unroll(dimension + 1, srcData, dstData, srcEnd, dstEnd, dataTypeSize);

                srcData += m_SrcStrides[dimension] * dataTypeSize;
                dstData += m_DstStrides[dimension] * dataTypeSize;
            }
        }
    }

    armnn::TensorShape m_SrcShape;
    std::array<size_type, armnn::MaxNumOfTensorDimensions> m_SrcStrides;
    std::array<size_type, armnn::MaxNumOfTensorDimensions> m_DstStrides;
};

} // namespace

namespace armnnUtils
{

armnn::TensorShape TransposeTensorShape(const armnn::TensorShape& srcShape, const armnn::PermutationVector& mappings)
{
    assert(srcShape.GetNumDimensions() == mappings.GetSize());

    const unsigned int numDims = mappings.GetSize();
    unsigned int outDims[armnn::MaxNumOfTensorDimensions];

    for (unsigned int i = 0U; i < numDims; ++i)
    {
        outDims[i] = srcShape[mappings[i]];
    }
    armnn::TensorShape permutedShape(numDims, outDims);
    return permutedShape;
}

armnn::TensorInfo TransposeTensorShape(const armnn::TensorInfo& info, const armnn::PermutationVector& mappings)
{
    armnn::TensorInfo outInfo(info);
    outInfo.SetShape(TransposeTensorShape(info.GetShape(), mappings));
    return outInfo;
}

void Transpose(const armnn::TensorShape& srcShape, const armnn::PermutationVector& mappings,
             const void* src, void* dst, size_t dataTypeSize)
{
    TransposeLoop(srcShape, mappings).Unroll(src, dst, dataTypeSize);
}

} // namespace armnnUtils
