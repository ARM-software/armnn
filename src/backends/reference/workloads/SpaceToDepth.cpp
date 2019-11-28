//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SpaceToDepth.hpp"

#include <armnnUtils/DataLayoutIndexed.hpp>

using namespace armnnUtils;

namespace {
    unsigned int GetOffset(const armnn::TensorShape& shape,
        unsigned int c,
        unsigned int h,
        unsigned int w,
        unsigned int b,
        const DataLayoutIndexed& dataLayout)
    {
        if (dataLayout.GetDataLayout() == armnn::DataLayout::NHWC)
        {
            return ((b * shape[dataLayout.GetHeightIndex()] + h) * shape[dataLayout.GetWidthIndex()] + w) *
                shape[dataLayout.GetChannelsIndex()] + c;
        }
        else
        {
            return ((b * shape[dataLayout.GetChannelsIndex()] + c) * shape[dataLayout.GetHeightIndex()] + h) *
                shape[dataLayout.GetWidthIndex()] + w;
        }
    }
}

namespace armnn
{

void SpaceToDepth(const TensorInfo& inputInfo,
                  const TensorInfo& outputInfo,
                  const SpaceToDepthDescriptor& params,
                  Decoder<float>& inputData,
                  Encoder<float>& outputData)
{
    DataLayoutIndexed dataLayout = params.m_DataLayout;

    const TensorShape& inputShape = inputInfo.GetShape();
    const TensorShape& outputShape = outputInfo.GetShape();

    const unsigned int inputBatchSize = inputShape[0];
    const unsigned int inputChannels = inputShape[dataLayout.GetChannelsIndex()];

    const unsigned int outputHeight = outputShape[dataLayout.GetHeightIndex()];
    const unsigned int outputWidth = outputShape[dataLayout.GetWidthIndex()];
    const unsigned int outputChannels = outputShape[dataLayout.GetChannelsIndex()];

    const unsigned int blockSize = params.m_BlockSize;

    if (blockSize == 0)
    {
        throw InvalidArgumentException(
            "Input shape must be divisible by block size in all spatial dimensions: Block size is"
            " equal to zero");
    }

    for (unsigned int outChannelIndex = 0; outChannelIndex < outputChannels; outChannelIndex++)
    {
        unsigned int inChannelIndex = outChannelIndex % inputChannels;

        unsigned int shiftW = (outChannelIndex / inputChannels) % blockSize;
        unsigned int shiftH = (outChannelIndex / inputChannels) / blockSize;

        for (unsigned int outH = 0; outH < outputHeight; outH++)
        {
            for (unsigned int outW = 0; outW < outputWidth; outW++)
            {
                for (unsigned int inBatchIndex = 0; inBatchIndex < inputBatchSize; inBatchIndex++)
                {
                    unsigned int inOffset = GetOffset(inputShape,
                        inChannelIndex,
                        (outH * blockSize + shiftH),
                        (outW * blockSize + shiftW),
                        inBatchIndex,
                        dataLayout);

                    unsigned int outOffset = GetOffset(outputShape,
                        outChannelIndex,
                        outH,
                        outW,
                        inBatchIndex,
                        dataLayout);

                    outputData += outOffset;
                    inputData += inOffset;
                    outputData.Set(inputData.Get());
                    inputData -= inOffset;
                    outputData -= outOffset;
                }
            }
        }
    }
}

void SpaceToDepth(const TensorInfo& inputInfo,
    const TensorInfo& outputInfo,
    const SpaceToDepthDescriptor& params,
    Decoder<float>& inputData,
    Encoder<float>& outData);

} //namespace armnn
