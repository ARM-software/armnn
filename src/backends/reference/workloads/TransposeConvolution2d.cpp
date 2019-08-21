//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TransposeConvolution2d.hpp"

#include <DataLayoutIndexed.hpp>

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

    unsigned int numBatches = inputShape[0];

    unsigned int inputWidth  = inputShape[widthIndex];
    unsigned int inputHeight = inputShape[heightIndex];
    unsigned int inputDepth  = inputShape[channelsIndex];

    unsigned int weightsHeight = weightsShape[heightIndex];
    unsigned int weightsWidth  = weightsShape[widthIndex];

    unsigned int outputHeight = outputShape[heightIndex];
    unsigned int outputWidth  = outputShape[widthIndex];
    unsigned int outputDepth  = outputShape[channelsIndex];

    unsigned int paddingLeft = descriptor.m_PadLeft;
    unsigned int paddingTop  = descriptor.m_PadTop;

    unsigned int strideX = descriptor.m_StrideX;
    unsigned int strideY = descriptor.m_StrideY;

    std::vector<float> outputBuffer(outputShape.GetNumElements(), 0);

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
                                    const unsigned int inputIndex =
                                        dataLayoutIndexed.GetIndex(inputShape, batch, dInput, yInput, xInput);
                                    inputDecoder[inputIndex];

                                    const unsigned int weightsIndex =
                                        dataLayoutIndexed.GetIndex(weightsShape, dOutput, dInput, yWeights, xWeights);
                                    weightsDecoder[weightsIndex];

                                    const unsigned int outputIndex =
                                        dataLayoutIndexed.GetIndex(outputShape, batch, dOutput, yOutput, xOutput);
                                    outputEncoder[outputIndex];

                                    float output = outputBuffer[outputIndex];
                                    output += inputDecoder.Get() * weightsDecoder.Get();
                                    outputBuffer[outputIndex] = output;
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
