//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchToSpaceNd.hpp"

#include "RefWorkloadUtils.hpp"

#include <armnn/Types.hpp>

#include <armnn/utility/Assert.hpp>

using namespace armnnUtils;

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
                    const std::vector<std::pair<unsigned int, unsigned int>>& cropsData,
                    Decoder<float>& inputDecoder,
                    Encoder<float>& outputEncoder)
{
    TensorShape inputShape = inputTensorInfo.GetShape();

    ARMNN_ASSERT_MSG(inputShape.GetNumDimensions() == 4, "Expected Input with 4 Dimensions");

    TensorShape outputShape = outputTensorInfo.GetShape();

    ARMNN_ASSERT_MSG(outputShape.GetNumDimensions() == 4, "Expected Output with 4 Dimensions");

    const unsigned int inputBatchSize = inputShape[0];
    const unsigned int channels = inputShape[dataLayout.GetChannelsIndex()];

    const unsigned int outputBatchSize = outputShape[0];
    const unsigned int outputHeight = outputShape[dataLayout.GetHeightIndex()];
    const unsigned int outputWidth = outputShape[dataLayout.GetWidthIndex()];

    ARMNN_ASSERT_MSG(blockShape.size() > 0, "BlockShape must contain 1 or more entries");

    const unsigned int blockShapeHeight = blockShape[0];
    const unsigned int blockShapeWidth = blockShape[1];

    ARMNN_ASSERT_MSG(cropsData.size() > 0, "Crops must contain 1 or more entries");

    const unsigned int cropsTop = cropsData[0].first;
    const unsigned int cropsLeft = cropsData[1].first;

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

                    outputEncoder[outOffset];
                    inputDecoder[inOffset];
                    outputEncoder.Set(inputDecoder.Get());
                }
            }
        }
    }
}

} //namespace armnn
