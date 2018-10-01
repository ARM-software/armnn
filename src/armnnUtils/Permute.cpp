//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Permute.hpp"

#include "Half.hpp"
#include <armnn/Tensor.hpp>

#include <cassert>

namespace
{

class PermuteLoop
{
public:
    using size_type = unsigned int;

    PermuteLoop(const armnn::TensorShape& dstShape, const armnn::PermutationVector& mappings)
        : m_DstShape(dstShape)
    {
        assert(dstShape.GetNumDimensions() == mappings.GetSize());

        const size_type numDims = dstShape.GetNumDimensions();

        size_type srcStride = 1U;
        size_type dstStride = 1U;

        for (size_type i = numDims - 1U, k = 0U; k < numDims; ++k, --i)
        {
            m_SrcStrides[mappings[i]] = srcStride;
            m_DstStrides[i] = dstStride;

            srcStride *= dstShape[mappings[i]];
            dstStride *= dstShape[i];
        }
    }

    template <typename T>
    void Unroll(const T* srcData, T* dstData)
    {
        const T* const srcEnd = srcData + m_DstShape.GetNumElements();
        T* const       dstEnd = dstData + m_DstShape.GetNumElements();
        Unroll(0, srcData, dstData, srcEnd, dstEnd);
    }

private:
    template <typename T>
    void Unroll(size_type dimension, const T* srcData, T* dstData, const T* srcEnd, T* dstEnd)
    {
        assert(srcData < srcEnd);
        assert(dstData < dstEnd);

        if (dimension >= m_DstShape.GetNumDimensions())
        {
            *dstData = *srcData;
        }
        else
        {
            for (size_type i = 0; i < m_DstShape[dimension]; i++)
            {
                Unroll(dimension + 1, srcData, dstData, srcEnd, dstEnd);

                srcData += m_SrcStrides[dimension];
                dstData += m_DstStrides[dimension];
            }
        }
    }

    armnn::TensorShape m_DstShape;
    std::array<size_type, armnn::MaxNumOfTensorDimensions> m_SrcStrides;
    std::array<size_type, armnn::MaxNumOfTensorDimensions> m_DstStrides;
};

} // namespace

namespace armnnUtils
{

armnn::TensorShape Permuted(const armnn::TensorShape& srcShape, const armnn::PermutationVector& mappings)
{
    assert(srcShape.GetNumDimensions() == mappings.GetSize());

    const unsigned int numDims = mappings.GetSize();
    unsigned int outDims[armnn::MaxNumOfTensorDimensions];

    for (unsigned int i = 0U; i < numDims; ++i)
    {
        outDims[mappings[i]] = srcShape[i];
    }

    armnn::TensorShape permutedShape(numDims, outDims);
    return permutedShape;
}

armnn::TensorInfo Permuted(const armnn::TensorInfo& info, const armnn::PermutationVector& mappings)
{
    armnn::TensorInfo outInfo(info);
    outInfo.SetShape(Permuted(info.GetShape(), mappings));
    return outInfo;
}

template <typename T>
void Permute(const armnn::TensorShape& dstShape, const armnn::PermutationVector& mappings, const T* src, T* dst)
{
    PermuteLoop(dstShape, mappings).Unroll(src, dst);
}

// Instantiates for types.
template void Permute(const armnn::TensorShape& dstShape, const armnn::PermutationVector& mappings,
                      const armnn::Half* src, armnn::Half* dst);
template void Permute(const armnn::TensorShape& dstShape, const armnn::PermutationVector& mappings,
                      const float* src, float* dst);
template void Permute(const armnn::TensorShape& dstShape, const armnn::PermutationVector& mappings,
                      const uint8_t* src, uint8_t* dst);
template void Permute(const armnn::TensorShape& dstShape, const armnn::PermutationVector& mappings,
                      const int32_t* src, int32_t* dst);

} // namespace armnnUtils
