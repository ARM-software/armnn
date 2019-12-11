//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CpuTensorHandle.hpp"
#include "ITensorHandle.hpp"

#include <armnn/Tensor.hpp>

#include <Half.hpp>
#include <Permute.hpp>
#include <Profiling.hpp>

#include <boost/cast.hpp>

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
    static_assert(MaxNumOfTensorDimensions == 5, "Please update CopyTensorContents");

    TensorShape srcStrides      = srcTensor->GetStrides();
    const TensorShape& srcShape = srcTensor->GetShape();
    TensorShape dstStrides      = dstTensor->GetStrides();
    const TensorShape& dstShape = dstTensor->GetShape();

    size_t srcDepth    = 1;
    size_t srcBatches  = 1;
    size_t srcChannels = 1;
    size_t srcHeight   = 1;
    size_t srcWidth    = 1;
    AssignValues(srcShape.GetNumDimensions(),
                 0,
                 srcShape,
                 srcWidth,
                 srcHeight,
                 srcChannels,
                 srcBatches,
                 srcDepth);

    size_t srcDepthStride   = 0;
    size_t srcBatchStride   = 0;
    size_t srcChannelStride = 0;
    size_t srcHeightStride  = 0;
    size_t srcWidthStride   = 0;
    AssignValues(srcStrides.GetNumDimensions(),
                 0,
                 srcStrides,
                 srcWidthStride,
                 srcHeightStride,
                 srcChannelStride,
                 srcBatchStride,
                 srcDepthStride);

    size_t dstDepth    = 1;
    size_t dstBatches  = 1;
    size_t dstChannels = 1;
    size_t dstHeight   = 1;
    size_t dstWidth    = 1;
    AssignValues(dstShape.GetNumDimensions(),
                 0,
                 dstShape,
                 dstWidth,
                 dstHeight,
                 dstChannels,
                 dstBatches,
                 dstDepth);

    size_t dstDepthStride   = 0;
    size_t dstBatchStride   = 0;
    size_t dstChannelStride = 0;
    size_t dstHeightStride  = 0;
    size_t dstWidthStride   = 0;
    AssignValues(dstStrides.GetNumDimensions(),
                 0,
                 dstStrides,
                 dstWidthStride,
                 dstHeightStride,
                 dstChannelStride,
                 dstBatchStride,
                 dstDepthStride);

    const unsigned char* srcData;
    unsigned char* dstData;
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Synchronize buffers");
        srcData = static_cast<const uint8_t*>(srcTensor->Map());
        dstData = static_cast<uint8_t*>(dstTensor->Map());
    }

    size_t copyLength   = std::min(srcWidth * srcWidthStride, dstWidth * dstWidthStride);
    size_t copyHeight   = std::min(srcHeight, dstHeight);
    size_t copyChannels = std::min(srcChannels, dstChannels);
    size_t copyBatches  = std::min(srcBatches, dstBatches);
    size_t copyDepth    = std::min(srcDepth, dstDepth);

    for (unsigned int d = 0; d < copyDepth; ++d)
    {
        auto srcPtrDepth = srcData;
        auto dstPtrDepth = dstData;
        for (unsigned int b = 0; b < copyBatches; ++b)
        {
            auto srcPtrBatch = srcData;
            auto dstPtrBatch = dstData;
            for (unsigned int c = 0; c < copyChannels; ++c)
            {
                auto srcPtrChannel = srcData;
                auto dstPtrChannel = dstData;
                for (unsigned int h = 0; h < copyHeight; ++h)
                {
                    copy(dstData, srcData, copyLength);
                    dstData += dstHeightStride;
                    srcData += srcHeightStride;
                }
                dstData += (static_cast<long>(dstChannelStride) - (dstData - dstPtrChannel));
                srcData += (static_cast<long>(srcChannelStride) - (srcData - srcPtrChannel));
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
            boost::polymorphic_downcast<SrcTensorHandleType*>(descriptor.m_Inputs[i]);
        DstTensorHandleType* const dstTensorHandle =
            boost::polymorphic_downcast<DstTensorHandleType*>(descriptor.m_Outputs[i]);

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
