//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TransposeConvolution2d.hpp"

#include <armnnUtils/DataLayoutIndexed.hpp>

namespace armnn
{

using namespace armnnUtils;

void TransposeConvolution2dImpl(const TransposeConvolution2dDescriptor& descriptor,
                                const TensorShape& inputShape,
                                Decoder<float>& inputDecoder,
                                const TensorShape& outputShape,
                                Encoder<float>& outputEncoder,
                                const TensorShape& weightsShape,
                                Decoder<float>& weightsDecoder,
                                Decoder<float>* biasesDecoder)
{
    if (descriptor.m_BiasEnabled && !biasesDecoder)
    {
        throw InvalidArgumentException("Biases enabled but no bias data provided");
    }
    const DataLayoutIndexed dataLayoutIndexed(descriptor.m_DataLayout);
    const unsigned int channelsIndex = dataLayoutIndexed.GetChannelsIndex();
    const unsigned int heightIndex   = dataLayoutIndexed.GetHeightIndex();
    const unsigned int widthIndex    = dataLayoutIndexed.GetWidthIndex();

    const unsigned int numBatches = inputShape[0];

    const unsigned int inputWidth  = inputShape[widthIndex];
    const unsigned int inputHeight = inputShape[heightIndex];
    const unsigned int inputDepth  = inputShape[channelsIndex];

    const unsigned int weightsHeight = weightsShape[heightIndex];
    const unsigned int weightsWidth  = weightsShape[widthIndex];
    const unsigned int weightsDepth  = weightsShape[channelsIndex];

    const unsigned int outputHeight = outputShape[heightIndex];
    const unsigned int outputWidth  = outputShape[widthIndex];
    const unsigned int outputDepth  = outputShape[channelsIndex];

    const unsigned int paddingLeft = descriptor.m_PadLeft;
    const unsigned int paddingTop  = descriptor.m_PadTop;

    const unsigned int strideX = descriptor.m_StrideX;
    const unsigned int strideY = descriptor.m_StrideY;

    std::vector<float> outputBuffer(outputShape.GetNumElements(), 0);

    const std::vector<float> inputVec = inputDecoder.DecodeTensor(inputShape);
    const std::vector<float> filterVec = weightsDecoder.DecodeTensor(weightsShape);

    for (unsigned int batch = 0u; batch < numBatches; ++batch)
    {
        for (unsigned int yInput = 0u; yInput < inputHeight; ++yInput)
        {
            for (unsigned int xInput = 0u; xInput < inputWidth; ++xInput)
            {
                unsigned int xOutputOrigin = xInput * strideX - paddingLeft;
                unsigned int yOutputOrigin = yInput * strideY - paddingTop;

                for (unsigned int dOutput = 0u; dOutput < outputDepth; ++dOutput)
                {
                    for (unsigned int yWeights = 0u; yWeights < weightsHeight; ++yWeights)
                    {
                        for (unsigned int xWeights = 0u; xWeights < weightsWidth; ++xWeights)
                        {
                            unsigned int yOutput = yOutputOrigin + yWeights;
                            unsigned int xOutput = xOutputOrigin + xWeights;

                            if (yOutput < outputHeight && xOutput< outputWidth)
                            {
                                for (unsigned int dInput = 0u; dInput < inputDepth; dInput++)
                                {
                                    unsigned int inputIndex;
                                    unsigned int outputIndex;
                                    unsigned int weightsIndex;

                                    if(descriptor.m_DataLayout == armnn::DataLayout::NHWC)
                                    {
                                        inputIndex   = batch  * inputHeight * inputWidth * inputDepth +
                                                       yInput * inputWidth * inputDepth +
                                                       xInput * inputDepth +
                                                       dInput;

                                        weightsIndex = dOutput  * weightsHeight * weightsWidth * weightsDepth +
                                                       yWeights * weightsWidth * weightsDepth +
                                                       xWeights * weightsDepth +
                                                       dInput;

                                        outputIndex  = batch   * outputHeight * outputWidth * outputDepth +
                                                       yOutput * outputWidth * outputDepth +
                                                       xOutput * outputDepth +
                                                       dOutput;
                                    }
                                    else
                                    {
                                        inputIndex   = batch  * inputDepth * inputHeight * inputWidth +
                                                       dInput * inputHeight * inputWidth +
                                                       yInput * inputWidth +
                                                       xInput;

                                        weightsIndex = dOutput  * weightsDepth * weightsHeight * weightsWidth +
                                                       dInput   * weightsHeight * weightsWidth +
                                                       yWeights * weightsWidth +
                                                       xWeights;

                                        outputIndex  = batch   * outputDepth * outputHeight * outputWidth +
                                                       dOutput * outputHeight * outputWidth +
                                                       yOutput * outputWidth +
                                                       xOutput;
                                    }

                                    outputBuffer[outputIndex] += inputVec[inputIndex] * filterVec[weightsIndex];
                                }
                            }
                        }
                    }

                }
            }
        }
    }

    // Apply bias (if enabled)
    if (descriptor.m_BiasEnabled)
    {
        outputEncoder[0];
        Decoder<float>& rBiasesDecoder = *biasesDecoder;

        for (unsigned int batch = 0u; batch < numBatches; ++batch)
        {
            for (unsigned int dOutput = 0u; dOutput < outputDepth; ++dOutput)
            {
                rBiasesDecoder[dOutput];
                for (unsigned int yOutput = 0u; yOutput < outputHeight; ++yOutput)
                {
                    for (unsigned int xOutput = 0u; xOutput < outputWidth; ++xOutput)
                    {
                        const unsigned int outputIndex =
                            dataLayoutIndexed.GetIndex(outputShape, batch, dOutput, yOutput, xOutput);
                        outputBuffer[outputIndex] += rBiasesDecoder.Get();
                    }
                }
            }
        }
    }
    outputEncoder[0];
    for (float output : outputBuffer)
    {
        outputEncoder.Set(output);
        ++outputEncoder;
    }
}

} // namespace armnn
