//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CpuTensorHandle.hpp"

#include <armnn/backends/ITensorHandle.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnnUtils/Permute.hpp>

#include <Half.hpp>
#include <Profiling.hpp>


namespace armnn
{
namespace
{

template <typename ArrayType, typename Arg>
void AssignValues(unsigned int num, unsigned int& idx, const ArrayType& array, Arg& arg)
{
    if (idx >= num)
    {
        return;
    }

    arg = array[(num - 1) - idx];
    idx++;
}

template <typename T, typename ArrayType, typename... Args>
void AssignValues(unsigned int num, unsigned int idx, const ArrayType& array, T& assignee, Args&... args)
{
    AssignValues(num, idx, array, assignee);

    AssignValues(num, idx, array, args...);
}

}    // anonymous namespace

template <typename CopyFunc>
void CopyTensorContentsGeneric(const ITensorHandle* srcTensor, ITensorHandle* dstTensor, CopyFunc copy)
{
    // For ease of understanding, names are assigned to the dimensions
    // of the tensor as if NHWC, however this routine works with any 5D tensor
    static_assert(MaxNumOfTensorDimensions == 5, "Please update CopyTensorContents");

    TensorShape srcStrides      = srcTensor->GetStrides();
    const TensorShape& srcShape = srcTensor->GetShape();
    const auto srcSize          = srcTensor->GetStrides()[0] * srcShape[0];
    IgnoreUnused(srcSize);  // Only used for asserts
    TensorShape dstStrides      = dstTensor->GetStrides();
    const TensorShape& dstShape = dstTensor->GetShape();
    const auto dstSize          = dstTensor->GetStrides()[0] * dstShape[0];
    IgnoreUnused(dstSize);  // Only used for asserts

    size_t srcDepth    = 1;
    size_t srcBatches  = 1;
    size_t srcHeight   = 1;
    size_t srcWidth    = 1;
    size_t srcChannels = 1;
    AssignValues(srcShape.GetNumDimensions(),
                 0,
                 srcShape,
                 srcChannels,
                 srcWidth,
                 srcHeight,
                 srcBatches,
                 srcDepth);

    size_t srcDepthStride   = 0;
    size_t srcBatchStride   = 0;
    size_t srcHeightStride  = 0;
    size_t srcWidthStride   = 0;
    size_t srcChannelStride = 0;
    AssignValues(srcStrides.GetNumDimensions(),
                 0,
                 srcStrides,
                 srcChannelStride,
                 srcWidthStride,
                 srcHeightStride,
                 srcBatchStride,
                 srcDepthStride);

    size_t dstDepth    = 1;
    size_t dstBatches  = 1;
    size_t dstHeight   = 1;
    size_t dstWidth    = 1;
    size_t dstChannels = 1;
    AssignValues(dstShape.GetNumDimensions(),
                 0,
                 dstShape,
                 dstChannels,
                 dstWidth,
                 dstHeight,
                 dstBatches,
                 dstDepth);

    size_t dstDepthStride   = 0;
    size_t dstBatchStride   = 0;
    size_t dstHeightStride  = 0;
    size_t dstWidthStride   = 0;
    size_t dstChannelStride = 0;
    AssignValues(dstStrides.GetNumDimensions(),
                 0,
                 dstStrides,
                 dstChannelStride,
                 dstWidthStride,
                 dstHeightStride,
                 dstBatchStride,
                 dstDepthStride);

    const unsigned char* srcDataStart;
    unsigned char* dstDataStart;
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Synchronize buffers");
        srcDataStart = static_cast<const uint8_t*>(srcTensor->Map());
        dstDataStart = static_cast<uint8_t*>(dstTensor->Map());
    }

    size_t copyLength  = std::min(srcChannels * srcChannelStride, dstChannels * dstChannelStride);
    size_t copyWidth   = std::min(srcWidth, dstWidth);
    size_t copyHeight  = std::min(srcHeight, dstHeight);
    size_t copyBatches = std::min(srcBatches, dstBatches);
    size_t copyDepth   = std::min(srcDepth, dstDepth);

    // Coalesce inner dimensions where possible
    // to reduce overheard calling copy() and to
    // allow for memory bandwidth optimisations
    if (copyLength == srcWidthStride &&
        copyLength == dstWidthStride)
    {
        // There is no special padding between rows,
        // and sizes are compatible, so copy whole rows
        copyLength *= copyWidth;
        copyWidth = 1;

        if (copyLength == srcHeightStride &&
            copyLength == dstHeightStride)
        {
            // There is no special padding between batches
            // and sizes are compatible so copy whole batches
            copyLength *= copyHeight;
            copyHeight = 1;
        }
    }

    const unsigned char* srcData = srcDataStart;
    unsigned char* dstData = dstDataStart;
    for (unsigned int d = 0; d < copyDepth; ++d)
    {
        auto srcPtrDepth = srcData;
        auto dstPtrDepth = dstData;
        for (unsigned int b = 0; b < copyBatches; ++b)
        {
            auto srcPtrBatch = srcData;
            auto dstPtrBatch = dstData;
            for (unsigned int h = 0; h < copyHeight; ++h)
            {
                auto srcPtrChannel = srcData;
                auto dstPtrChannel = dstData;
                for (unsigned int w = 0; w < copyWidth; ++w)
                {
                    ARMNN_ASSERT(srcData >= srcDataStart && srcData + copyLength <= srcDataStart + srcSize);
                    ARMNN_ASSERT(dstData >= dstDataStart && dstData + copyLength <= dstDataStart + dstSize);
                    copy(dstData, srcData, copyLength);
                    dstData += dstWidthStride;
                    srcData += srcWidthStride;
                }
                dstData += (static_cast<long>(dstHeightStride) - (dstData - dstPtrChannel));
                srcData += (static_cast<long>(srcHeightStride) - (srcData - srcPtrChannel));
            }
            dstData += (static_cast<long>(dstBatchStride) - (dstData - dstPtrBatch));
            srcData += (static_cast<long>(srcBatchStride) - (srcData - srcPtrBatch));
        }
        dstData += (static_cast<long>(dstDepthStride) - (dstData - dstPtrDepth));
        srcData += (static_cast<long>(srcDepthStride) - (srcData - srcPtrDepth));
    }

    srcTensor->Unmap();
    dstTensor->Unmap();
}

template <typename SrcTensorHandleType, typename DstTensorHandleType, typename DescriptorType>
void GatherTensorHandlePairs(const DescriptorType& descriptor,
                             std::vector<std::pair<SrcTensorHandleType*, DstTensorHandleType*>>& tensorHandlePairs)
{
    const unsigned int numInputs = static_cast<unsigned int>(descriptor.m_Inputs.size());
    tensorHandlePairs.reserve(numInputs);

    for (unsigned int i = 0; i < numInputs; ++i)
    {
        SrcTensorHandleType* const srcTensorHandle =
            PolymorphicDowncast<SrcTensorHandleType*>(descriptor.m_Inputs[i]);
        DstTensorHandleType* const dstTensorHandle =
            PolymorphicDowncast<DstTensorHandleType*>(descriptor.m_Outputs[i]);

        tensorHandlePairs.emplace_back(srcTensorHandle, dstTensorHandle);
    }
}

int32_t ConvertMaskToACLFormat(int32_t mask, int32_t numDim);

armnn::ConstTensor PermuteTensor(const ConstCpuTensorHandle* tensor,
                                 const PermutationVector& permutationVector,
                                 void* permuteBuffer);

void ReshapeWeightsForAcl(TensorInfo& weightInfo, DataLayout dataLayout);

TensorInfo ConvertWeightTensorInfoFromArmnnToAcl(const TensorInfo& weightInfo, DataLayout dataLayout);

armnn::ConstTensor ConvertWeightTensorFromArmnnToAcl(const ConstCpuTensorHandle* weightTensor,
                                                     DataLayout dataLayout,
                                                     void* permuteBuffer);

}  //namespace armnn
