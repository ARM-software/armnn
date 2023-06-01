//
// Copyright © 2017-2020,2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchToSpaceNd.hpp"

#include <armnnUtils/DataLayoutIndexed.hpp>

using namespace armnnUtils;

namespace armnn
{

unsigned int Offset(const TensorShape& shape,
                    unsigned int batch,
                    unsigned int height,
                    unsigned int width,
                    unsigned int channels,
                    const DataLayoutIndexed& dataLayout)
{
    // 3D Tensors
    unsigned int channelDimension3D = dataLayout.GetDataLayout() == DataLayout::NCHW ? 1 : 2;
    if (shape.GetNumDimensions() == 3)
    {
        return (batch * shape[dataLayout.GetHeightIndex()] + height) * shape[channelDimension3D] + channels;
    }
    // 4D Tensors
    else if (shape.GetNumDimensions() == 4)
    {
        if (dataLayout.GetDataLayout() == DataLayout::NHWC)
        {
            return ((batch * shape[dataLayout.GetHeightIndex()] + height) *
                    shape[dataLayout.GetWidthIndex()] + width) *
                    shape[dataLayout.GetChannelsIndex()] + channels;
        }
        else
        {
            return ((batch * shape[dataLayout.GetChannelsIndex()] + channels) *
                    shape[dataLayout.GetHeightIndex()] + height) *
                    shape[dataLayout.GetWidthIndex()] + width;
        }
    }
    else
    {
        throw InvalidArgumentException("Tensor rank must be either 3 or 4", CHECK_LOCATION());
    }
}

void BatchToSpaceNd(const TensorInfo& inputInfo,
                    const TensorInfo& outputInfo,
                    const BatchToSpaceNdDescriptor& params,
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

    TensorShape inputShape = inputInfo.GetShape();
    TensorShape outputShape = outputInfo.GetShape();

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

    const unsigned int cropsTop  = params.m_Crops[0].first;
    const unsigned int cropsLeft = (rank == 3) ? 0 : params.m_Crops[1].first;

    for (unsigned int inBatch = 0; inBatch < inputBatchSize; ++inBatch)
    {
        const unsigned int outBatch = inBatch % outputBatchSize;
        const unsigned int spatialOffset = inBatch / outputBatchSize;

        for (unsigned int inH = 0; inH < inputHeight; ++inH)
        {
            const unsigned int outH = inH * blockHeight + spatialOffset / blockWidth - cropsTop;

            if (outH >= outputHeight)
            {
                continue;
            }

            for (unsigned int inW = 0; inW < inputWidth; ++inW)
            {
                const unsigned int outW = inW * blockWidth + spatialOffset % blockWidth - cropsLeft;

                if (outW >= outputWidth)
                {
                    continue;
                }

                for (unsigned int c = 0; c < channels; c++)
                {
                    unsigned int outOffset = Offset(outputShape, outBatch, outH, outW, c, dataLayout);
                    unsigned int inOffset = Offset(inputShape, inBatch, inH, inW, c, dataLayout);

                    outputData[outOffset];
                    inputData[inOffset];
                    outputData.Set(inputData.Get());
                }
            }
        }
    }
}

} //namespace armnn
