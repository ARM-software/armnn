//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchToSpaceNd.hpp"

#include "RefWorkloadUtils.hpp"

#include <armnn/Types.hpp>

#include <boost/assert.hpp>

namespace armnn
{

inline unsigned int Offset(const TensorShape& shape, unsigned int batch, unsigned int height, unsigned int width,
        unsigned int channels, const DataLayoutIndexed& dataLayout)
{
    if (dataLayout.GetDataLayout() == DataLayout::NHWC)
    {
        return ((batch * shape[dataLayout.GetHeightIndex()] + height) * shape[dataLayout.GetWidthIndex()] + width) *
               shape[dataLayout.GetChannelsIndex()] + channels;
    }
    else
    {
        return ((batch * shape[dataLayout.GetChannelsIndex()] + channels) *
               shape[dataLayout.GetHeightIndex()] + height) *
               shape[dataLayout.GetWidthIndex()] + width;
    }
}

void BatchToSpaceNd(const DataLayoutIndexed& dataLayout,
                    const TensorInfo& inputTensorInfo,
                    const TensorInfo& outputTensorInfo,
                    const std::vector<unsigned int>& blockShape,
                    const std::vector<std::vector<unsigned int>>& cropsData,
                    const float* inputData,
                    float* outputData)
{
    TensorShape inputShape = inputTensorInfo.GetShape();
    unsigned int inputNumDims = inputShape.GetNumDimensions();
    if (inputNumDims != 4)
    {
        throw armnn::InvalidArgumentException("Expected Input with 4 Dimensions");
    }

    TensorShape outputShape = outputTensorInfo.GetShape();
    unsigned int outputNumDims = outputShape.GetNumDimensions();
    if (outputNumDims != 4)
    {
        throw armnn::InvalidArgumentException("Expected Output with 4 Dimensions");
    }

    const unsigned int inputBatchSize = inputShape[0];
    const unsigned int channels = inputShape[dataLayout.GetChannelsIndex()];

    const unsigned int outputBatchSize = outputShape[0];
    const unsigned int outputHeight = outputShape[dataLayout.GetHeightIndex()];
    const unsigned int outputWidth = outputShape[dataLayout.GetWidthIndex()];

    const unsigned int blockShapeHeight = blockShape[0];
    const unsigned int blockShapeWidth = blockShape[1];

    const unsigned int cropsTop = cropsData[0][0];
    const unsigned int cropsLeft = cropsData[1][0];

    for (unsigned int inBatch = 0; inBatch < inputBatchSize; ++inBatch)
    {
        const unsigned int outBatch = inBatch % outputBatchSize;
        const unsigned int spatialOffset = inBatch / outputBatchSize;

        for (unsigned int inH = 0; inH < inputTensorInfo.GetShape()[dataLayout.GetHeightIndex()]; ++inH) {
            const unsigned int outH = inH * blockShapeHeight + spatialOffset / blockShapeWidth - cropsTop;

            if (outH >= outputHeight)
            {
                continue;
            }

            for (unsigned int inW = 0; inW < inputTensorInfo.GetShape()[dataLayout.GetWidthIndex()]; ++inW) {
                const unsigned int outW = inW * blockShapeWidth + spatialOffset % blockShapeWidth - cropsLeft;

                if (outW >= outputWidth)
                {
                    continue;
                }

                for (unsigned int c = 0; c < channels; c++)
                {
                    unsigned int outOffset = Offset(outputShape, outBatch, outH, outW, c, dataLayout);
                    unsigned int inOffset = Offset(inputShape, inBatch, inH, inW, c, dataLayout);
                    outputData[outOffset] = inputData[inOffset];
                }
            }
        }
    }
}

} //namespace armnn
