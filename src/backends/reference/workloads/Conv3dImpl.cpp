//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Conv3dImpl.hpp"

namespace armnn
{

void Convolve3d(const TensorShape& rInputShape,
                Decoder<float>& rInputDecoder,
                const TensorShape& rOutputShape,
                Encoder<float>& rOutputEncoder,
                const TensorShape& rFilterShape,
                Decoder<float>& rFilterDecoder,
                bool biasEnabled,
                Decoder<float>* pBiasDecoder,
                DataLayout dataLayout,
                unsigned int paddingTop,
                unsigned int paddingLeft,
                unsigned int paddingFront,
                unsigned int xStride,
                unsigned int yStride,
                unsigned int zStride,
                unsigned int xDilation,
                unsigned int yDilation,
                unsigned int zDilation)
{
    if (biasEnabled && !pBiasDecoder)
    {
        throw InvalidArgumentException("Bias is enabled but the bias data is invalid");
    }
    const armnnUtils::DataLayoutIndexed dataLayoutIndexed(dataLayout);

    const unsigned int channelsIndex = dataLayoutIndexed.GetChannelsIndex();
    const unsigned int heightIndex   = dataLayoutIndexed.GetHeightIndex();
    const unsigned int widthIndex    = dataLayoutIndexed.GetWidthIndex();
    const unsigned int depthIndex    = dataLayoutIndexed.GetDepthIndex();

    const unsigned int inChannels   = rInputShape[channelsIndex];
    const unsigned int outChannels  = rOutputShape[channelsIndex];

    const unsigned int batchSize    = rOutputShape[0];
    const unsigned int outputHeight = rOutputShape[heightIndex];
    const unsigned int outputWidth  = rOutputShape[widthIndex];
    const unsigned int outputDepth  = rOutputShape[depthIndex];
    const unsigned int inputHeight  = rInputShape[heightIndex];
    const unsigned int inputWidth   = rInputShape[widthIndex];
    const unsigned int inputDepth   = rInputShape[depthIndex];

    // Conv3d weights layout: [D,H,W,I,O]
    const unsigned int filterDepth  = rFilterShape[0];
    const unsigned int filterHeight = rFilterShape[1];
    const unsigned int filterWidth  = rFilterShape[2];

    const std::vector<float> inputVec = rInputDecoder.DecodeTensor(rInputShape);
    const std::vector<float> filterVec = rFilterDecoder.DecodeTensor(rFilterShape);

    const TensorShape biasShape{outChannels};
    const std::vector<float> biasVec = biasEnabled ? pBiasDecoder->DecodeTensor(biasShape) : std::vector<float>();

    for (unsigned int batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        for (unsigned int zOutput = 0; zOutput < outputDepth; zOutput++)
        {
            for (unsigned int xOutput = 0; xOutput < outputWidth; xOutput++)
            {
                for (unsigned int yOutput = 0; yOutput < outputHeight; yOutput++)
                {
                    for (unsigned int cOutput = 0; cOutput < outChannels; cOutput++)
                    {
                        // This loop goes over each output element.
                        float sum = 0.0f;

                        // Loop over each input channel.
                        for (unsigned int zFilter = 0; zFilter < filterDepth; zFilter++)
                        {
                            for (unsigned int yFilter = 0; yFilter < filterHeight; yFilter++)
                            {
                                for (unsigned int xFilter = 0; xFilter < filterWidth; xFilter++)
                                {
                                    for (unsigned int cInput = 0; cInput < inChannels; cInput++)
                                    {
                                        // This loop goes over each input element for each output element.
                                        unsigned int filterIndex = 0;

                                        // Conv3d weights layout: [D,H,W,I,O]
                                        // Keep this implementation, as using DataLayoutIndexed::GetIndex
                                        // causes large performance regression.
                                        filterIndex = zFilter * filterHeight * filterWidth * inChannels * outChannels +
                                                      yFilter * filterWidth * inChannels * outChannels +
                                                      xFilter * inChannels * outChannels +
                                                      cInput * outChannels +
                                                      cOutput;

                                        unsigned int yInput = yOutput * yStride + yFilter * yDilation;
                                        unsigned int xInput = xOutput * xStride + xFilter * xDilation;
                                        unsigned int zInput = zOutput * zStride + zFilter * zDilation;

                                        float inputValue;

                                        // Check if we're in the padding.
                                        if (yInput < paddingTop || yInput >= inputHeight + paddingTop ||
                                            xInput < paddingLeft || xInput >= inputWidth + paddingLeft ||
                                            zInput < paddingFront || zInput >= inputDepth + paddingFront)
                                        {
                                            inputValue = 0.0f;
                                        }
                                        else
                                        {
                                            unsigned int inputIndex = 0;

                                            // Keep this implementation, as using DataLayoutIndexed::GetIndex
                                            // causes large performance regression.
                                            if (dataLayoutIndexed.GetDataLayout() == DataLayout::NDHWC)
                                            {
                                                inputIndex =
                                                        batchIdx * inputDepth * inputHeight * inputWidth * inChannels +
                                                        (zInput-paddingFront) * inputHeight * inputWidth * inChannels +
                                                        (yInput-paddingTop) * inputWidth * inChannels +
                                                        (xInput-paddingLeft) * inChannels +
                                                        cInput;
                                            }
                                            else
                                            {
                                                // NCDHW DataLayout
                                                inputIndex =
                                                        batchIdx * inputDepth * inputHeight * inputWidth * inChannels +
                                                        inputDepth * inputHeight * inputWidth * cInput +
                                                        (zInput-paddingFront) * inputHeight * inputWidth +
                                                        (yInput-paddingTop) * inputWidth +
                                                        xInput-paddingLeft;
                                            }

                                            inputValue = inputVec[inputIndex];
                                        }

                                        sum += filterVec[filterIndex] * inputValue;
                                    }
                                }
                            }
                        }

                        if (biasEnabled)
                        {
                            sum += biasVec[cOutput];
                        }

                        unsigned int outIdx;
                        if (dataLayoutIndexed.GetDataLayout() == DataLayout::NDHWC)
                        {
                            outIdx = batchIdx * outputDepth * outputHeight * outputWidth * outChannels +
                                     zOutput * outputHeight * outputWidth * outChannels +
                                     yOutput * outputWidth * outChannels +
                                     xOutput * outChannels +
                                     cOutput;
                        }
                        else
                        {
                            // NCDHW DataLayout
                            outIdx = batchIdx * outputDepth * outputHeight * outputWidth * outChannels +
                                     cOutput * outputDepth * outputHeight * outputWidth +
                                     zOutput * outputHeight * outputWidth +
                                     yOutput * outputWidth +
                                     xOutput;
                        }

                        rOutputEncoder[outIdx];
                        rOutputEncoder.Set(sum);
                    }
                }
            }
        }
    }
}

} // namespace armnn
