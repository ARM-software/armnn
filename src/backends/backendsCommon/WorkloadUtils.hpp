//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/ITensorHandle.hpp>
#include <armnn/backends/TensorHandle.hpp>
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

armnn::ConstTensor PermuteTensor(const ConstTensorHandle* tensor,
                                 const PermutationVector& permutationVector,
                                 void* permuteBuffer);

void ReshapeWeightsForAcl(TensorInfo& weightInfo, DataLayout dataLayout);

TensorInfo ConvertWeightTensorInfoFromArmnnToAcl(const TensorInfo& weightInfo, DataLayout dataLayout);

/// Weights for depthwise have a datalayout of [1,H,W,O] = [1,H,W,I*M]
/// This function coverts a TensorInfo from [1,H,W,I*M] to [1,I*M,H,W] (if NCHW) or keeps it at [1,H,W,I*M] (if NHWC)
/// as required by the compute library
/// Returns a tuple of converted weights tensor info and depth multiplier
std::tuple<TensorInfo, unsigned int> Convert1HWOTensorInfoToAcl(const TensorInfo& weightInfo,
                                                                const TensorInfo& inputInfo,
                                                                const DataLayout dataLayout);

armnn::ConstTensor ConvertWeightTensorFromArmnnToAcl(const ConstTensorHandle* weightTensor,
                                                     DataLayout dataLayout,
                                                     void* permuteBuffer);

/// Weights for depthwise have a datalayout of [1,H,W,O] = [1,H,W,I*M]
/// This function coverts a ConstCpuTensorHandle from [1,H,W,I*M] to [1,I*M,H,W] (if NCHW) or
/// keeps it at [1,H,W,I*M] (if NHWC) as required by the compute library
///
/// \param weightTensor - ConstTensorHandle of weights tensor
/// \param inputInfo - TensorInfo of input tensor
/// \param dataLayout - DataLayout of the input tensor
/// \param permuteBuffer - Pointer to memory with the size of tensor. Used for the permutation
/// \return tuple of transformed weights-ConstTensor and depthwise multiplier
std::tuple<ConstTensor, unsigned int> Convert1HWOTensorToAcl(const ConstTensorHandle* weightTensor,
                                                             const TensorInfo& inputInfo,
                                                             const DataLayout dataLayout,
                                                             void* permuteBuffer);

/// Converts a (weights) tensor from [1, H, W, I*M] = [1, H, W, O] to [M, I, H, W]
///
/// \param weightTensor - ConstTensorHandle of the weight tensor that should be converted
/// \param inputInfo - TensorInfo of the corresponding input tensor
/// \param dataLayout - DataLayout of the input tensor e.g. NHWC or NCHW
/// \param permuteBuffer - Memory location with the same size as the weight tensor to write converted data to
/// \return - A tuple of ConstTensor and unsigned int which is the converted weightTensor and the depthMultiplier
std::tuple<ConstTensor, unsigned int> Convert1HWOtoMIHW(const ConstTensorHandle* weightTensor,
                                                        const TensorInfo& inputInfo,
                                                        const DataLayout& dataLayout,
                                                        void* permuteBuffer);

/// Calculates the key index values needed for GatherNd: N, ND, K, W, C (N is always 1)
///
/// \param inputInfo0 - TensorInfo of the corresponding input tensor: params
/// \param inputInfo1 - TensorInfo of the corresponding input tensor: indices
/// \return - A map with names and values for  N, ND, K, W, C
std::map<std::string, unsigned int> CalculateGatherNdKeyIndices(TensorInfo inputInfo0, TensorInfo inputInfo1);

/// Generates a permutation vector of size rank that permutes the 2 most right dimensions
///
/// \param rank - Tensor rank, i.e. number of dimensions in the tensors
/// \return - A permutation vector that permutes the 2 last dimensions
armnn::PermutationVector GeneratePermutationVectorOnLastTwoDimensions(unsigned int rank);

}  //namespace armnn
