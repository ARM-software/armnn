//
// Copyright © 2017-2019,2023 Arm Ltd and Contributors. All rights reserved.
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
    // 3D Tensors
    unsigned int channelDimension3D = dataLayout.GetDataLayout() == DataLayout::NCHW ? 1 : 2;
    if (shape.GetNumDimensions() == 3)
    {
        return (b * shape[dataLayout.GetHeightIndex()] + h) * shape[channelDimension3D] + c;
    }
    // 4D Tensors
    else if (shape.GetNumDimensions() == 4)
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
    else
    {
        throw InvalidArgumentException("Tensor rank must be either 3 or 4", CHECK_LOCATION());
    }
}

void SpaceToBatchNd(const TensorInfo& inputInfo,
                    const TensorInfo& outputInfo,
                    const SpaceToBatchNdDescriptor& params,
                    Decoder<float>& inputData,
                    Encoder<float>& outputData)
{
    unsigned int rank = inputInfo.GetNumDimensions();
    if (rank != 3 && rank != 4 )
    {
        throw InvalidArgumentException("Tensor rank must be either 3 or 4, but it is " + std::to_string(rank),
                                       CHECK_LOCATION());
    }

    DataLayoutIndexed dataLayout = params.m_DataLayout;
    unsigned int channelDimension3D = params.m_DataLayout == DataLayout::NCHW ? 1 : 2;

    const TensorShape& inputShape = inputInfo.GetShape();
    const TensorShape& outputShape = outputInfo.GetShape();

    const unsigned int inputBatchSize  = inputShape[0];
    const unsigned int outputBatchSize = outputShape[0];

    const unsigned int channels = (rank == 3) ? inputShape[channelDimension3D]
                                              : inputShape[dataLayout.GetChannelsIndex()];

    const unsigned int inputHeight  = inputShape[dataLayout.GetHeightIndex()];
    const unsigned int inputWidth   = (rank == 3) ? 1 : inputShape[dataLayout.GetWidthIndex()];
    const unsigned int outputHeight = outputShape[dataLayout.GetHeightIndex()];
    const unsigned int outputWidth  = (rank == 3) ? 1 : outputShape[dataLayout.GetWidthIndex()];

    const unsigned int blockHeight = params.m_BlockShape[0];
    const unsigned int blockWidth  = (rank == 3) ? 1 : params.m_BlockShape[1];

    const unsigned int paddingTop  = params.m_PadList[0].first;
    const unsigned int paddingLeft = (rank == 3) ? 0 : params.m_PadList[1].first;

    for (unsigned int outB = 0; outB < outputBatchSize; ++outB)
    {
        unsigned int inB = outB % inputBatchSize;

        unsigned int shiftW = (outB / inputBatchSize) % blockWidth;
        unsigned int shiftH = (outB / inputBatchSize) / blockWidth;

        for (unsigned int outH = 0; outH < outputHeight; ++outH)
        {
            for (unsigned int outW = 0; outW < outputWidth; ++outW)
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

} //namespace armnn
