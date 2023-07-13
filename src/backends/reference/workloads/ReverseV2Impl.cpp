//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReverseV2Impl.hpp"

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/Logging.hpp>
#include <armnnUtils/Permute.hpp>

namespace armnn
{

// Get multi-dimensional index for input tensor
std::vector<unsigned int> ReverseGetMultIdx(const unsigned int idx,
                                            unsigned int inputRank,
                                            std::vector<unsigned int>& elementNumInner)
{
    std::vector<unsigned int> indexList(inputRank);

    unsigned int mIdx = idx;

    for (unsigned int iDim = 0; iDim < inputRank; ++iDim)
    {
        indexList[iDim] = static_cast<unsigned int>(mIdx / elementNumInner[iDim]);
        mIdx %= elementNumInner[iDim];
    }

    return indexList;
}

// Get flattened index for output encoder
unsigned int ReverseGetFlatIdx(const std::vector<unsigned int>& idxList,
                               unsigned int inputRank,
                               std::vector<unsigned int>& elementNumInner)
{
    unsigned int idx = 0;

    for (unsigned int iDim = 0; iDim < inputRank; ++iDim)
    {
        idx += idxList[iDim] * elementNumInner[iDim];
    }

    return idx;
}

// Relocate the coordinate to the reversed tensor
unsigned int ReverseRelocateIdx(unsigned int idx,
                                unsigned int inputRank,
                                std::vector<bool>& axisFlag,
                                std::vector<unsigned int>& dimSize,
                                std::vector<unsigned int>& elementNumInner)
{
    // Get the multidimensional index list for input
    auto inputIdxList = ReverseGetMultIdx(idx, inputRank, elementNumInner);

    std::vector<unsigned int> outputIdxList(inputRank);

    // Relocate the input index to the output one
    for (unsigned int iDim = 0; iDim < inputRank; ++iDim)
    {
        if (axisFlag[iDim])
        {
            outputIdxList[iDim] = dimSize[iDim] - inputIdxList[iDim] - 1;
        }
        else
        {
            outputIdxList[iDim] = inputIdxList[iDim];
        }
    }

    // Get the 1-dimensional flattened index for output
    unsigned int outputIdx = ReverseGetFlatIdx(outputIdxList, inputRank, elementNumInner);
    return outputIdx;
}

void ReverseV2(const TensorInfo& inputInfo,
               const TensorInfo& axisInfo,
               Decoder<float>& inputDecoder,
               Decoder<int>& axisDecoder,
               Encoder<float>& outputEncoder)
{
    unsigned int axesRank = static_cast<unsigned int>(axisInfo.GetNumElements());

    // Empty axis and empty tensor case: copy input to output
    if ((axesRank == 0) || inputInfo.GetNumElements() == 0)
    {
        for (unsigned idx = 0; idx < inputInfo.GetNumElements(); idx++)
        {
            float inputValue = inputDecoder.Get();
            inputDecoder += 1;
            outputEncoder.Set(inputValue);
            outputEncoder += 1;
        }
        return;
    }

    unsigned int inputRank = static_cast<unsigned int>(inputInfo.GetNumDimensions());

    std::vector<bool> axisFlag(inputRank, false);
    std::vector<unsigned int> dimSize(inputRank, 0);
    std::vector<int32_t> axis(axesRank, 0);

    // Decode the axis information
    for (unsigned int i=0; i < axesRank; i++)
    {
        axis[i] = axisDecoder.Get();
        axisDecoder += 1;
    }

    // Make sure the axes are positive
    for (int32_t axisElement: axis)
    {
        axisElement = axisElement < 0 ? axisElement + static_cast<int32_t>(inputRank) : axisElement;
        axisFlag[static_cast<uint32_t>(axisElement)] = true;
    }

    const TensorShape &inputShape = inputInfo.GetShape();

    unsigned int elementNum = inputInfo.GetNumElements();
    unsigned int baseDimSize = 1;

    std::vector<unsigned int> elementNumInner;

    // Get the number of element within the specific dimension
    for (unsigned int iDim = 0; iDim < inputRank; ++iDim) {
        dimSize[iDim] = inputShape[iDim];
        baseDimSize *= dimSize[iDim];
        elementNumInner.push_back(static_cast<unsigned int>(elementNum / baseDimSize));
    }

    // Iterate through all elements
    for (unsigned int idx = 0; idx < elementNum; ++idx)
    {
        float inputValue = inputDecoder.Get();
        inputDecoder += 1;
        auto outputIdx = ReverseRelocateIdx(idx, inputRank, axisFlag, dimSize, elementNumInner);
        outputEncoder[outputIdx];
        outputEncoder.Set(inputValue);
    }
}

} // namespace armnn