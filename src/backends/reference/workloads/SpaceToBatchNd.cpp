//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SpaceToBatchNd.hpp"

#include <armnnUtils/DataLayoutIndexed.hpp>

using namespace armnnUtils;

namespace armnn
{

unsigned int GetOffset(const TensorShape& shape,
                       unsigned int b,
                       unsigned int h,
                       unsigned int w,
                       unsigned int c,
                       const DataLayoutIndexed& dataLayout)
{
    if (dataLayout.GetDataLayout() == DataLayout::NHWC)
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

void SpaceToBatchNd(const TensorInfo& inputInfo,
                    const TensorInfo& outputInfo,
                    const SpaceToBatchNdDescriptor& params,
                    Decoder<float>& inputData,
                    Encoder<float>& outputData)
{
    DataLayoutIndexed dataLayout = params.m_DataLayout;

    const TensorShape& inputShape = inputInfo.GetShape();
    const TensorShape& outputShape = outputInfo.GetShape();

    const unsigned int channels = inputShape[dataLayout.GetChannelsIndex()];

    const unsigned int inputBatchSize = inputShape[0];
    const unsigned int inputHeight = inputShape[dataLayout.GetHeightIndex()];
    const unsigned int inputWidth = inputShape[dataLayout.GetWidthIndex()];

    const unsigned int outputBatchSize = outputShape[0];
    const unsigned int outputHeight = outputShape[dataLayout.GetHeightIndex()];
    const unsigned int outputWidth = outputShape[dataLayout.GetWidthIndex()];

    const unsigned int blockHeight = params.m_BlockShape[0];
    const unsigned int blockWidth = params.m_BlockShape[1];

    const unsigned int paddingTop = params.m_PadList[0].first;
    const unsigned int paddingLeft = params.m_PadList[1].first;

    for (unsigned int outB = 0; outB < outputBatchSize; outB++)
    {
        unsigned int inB = outB % inputBatchSize;

        unsigned int shiftW = (outB / inputBatchSize) % blockWidth;
        unsigned int shiftH = (outB / inputBatchSize) / blockWidth;

        for (unsigned int outH = 0; outH < outputHeight; outH++)
        {
            for (unsigned int outW = 0; outW < outputWidth; outW++)
            {
                if (outH * blockHeight + shiftH < paddingTop ||
                    outH * blockHeight + shiftH >= paddingTop + inputHeight ||
                    outW * blockWidth + shiftW < paddingLeft ||
                    outW * blockWidth + shiftW >= paddingLeft + inputWidth)
                {
                    for (unsigned int c = 0; c < channels; c++)
                    {
                        unsigned int outOffset = GetOffset(outputShape,
                                                           outB,
                                                           outH,
                                                           outW,
                                                           c,
                                                           dataLayout);
                        outputData += outOffset;
                        outputData.Set(0);
                        outputData -= outOffset;
                    }
                }
                else
                {
                    for (unsigned int c = 0; c < channels; c++)
                    {
                        unsigned int inOffset = GetOffset(inputShape,
                                                          inB,
                                                          (outH * blockHeight + shiftH) - paddingTop,
                                                          (outW * blockWidth + shiftW) - paddingLeft,
                                                          c,
                                                          dataLayout);

                        unsigned int outOffset = GetOffset(outputShape,
                                                           outB,
                                                           outH,
                                                           outW,
                                                           c,
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
}

void SpaceToBatchNd(const TensorInfo& inputInfo,
                    const TensorInfo& outputInfo,
                    const SpaceToBatchNdDescriptor& params,
                    Decoder<float>& inputData,
                    Encoder<float>& outData);

} //namespace armnn
