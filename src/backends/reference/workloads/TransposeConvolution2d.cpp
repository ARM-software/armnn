//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TransposeConvolution2d.hpp"

#include <DataLayoutIndexed.hpp>

namespace armnn
{

using namespace armnnUtils;

struct TensorData
{
    TensorShape        shape;
    std::vector<float> data;
};

TensorData SetUpStridedInput(const TensorShape& inputShape,
                             Decoder<float>& inputDecoder,
                             const TransposeConvolution2dDescriptor& descriptor,
                             const DataLayoutIndexed& dataLayoutIndexed)
{
    const unsigned int cIndex = dataLayoutIndexed.GetChannelsIndex();
    const unsigned int hIndex = dataLayoutIndexed.GetHeightIndex();
    const unsigned int wIndex = dataLayoutIndexed.GetWidthIndex();

    const unsigned int batches  = inputShape[0];
    const unsigned int channels = inputShape[cIndex];

    const unsigned int wInput = inputShape[wIndex];
    const unsigned int hInput = inputShape[hIndex];

    const unsigned int wStridedInput = 1u + descriptor.m_StrideX * (wInput - 1);
    const unsigned int hStridedInput = 1u + descriptor.m_StrideY * (hInput - 1);

    TensorData stridedInput;
    stridedInput.data  = std::vector<float>(batches * channels * wStridedInput * hStridedInput, 0.0f);
    stridedInput.shape = TensorShape(4);

    stridedInput.shape[0]      = batches;
    stridedInput.shape[cIndex] = channels;
    stridedInput.shape[hIndex] = hStridedInput;
    stridedInput.shape[wIndex] = wStridedInput;

    // expand input data with strides
    for (unsigned int batchIdx = 0u; batchIdx < batches; ++batchIdx)
    {
        for (unsigned int cInput = 0u; cInput < channels; ++cInput)
        {
            for (unsigned int yInput = 0u, yStrided = 0u;
                 yInput < hInput && yStrided < hStridedInput;
                 ++yInput, yStrided += descriptor.m_StrideY)
            {
                for (unsigned int xInput = 0u, xStrided = 0u;
                     xInput < wInput && xStrided < wStridedInput;
                     ++xInput, xStrided += descriptor.m_StrideX)
                {
                    unsigned int inputIdx =
                        dataLayoutIndexed.GetIndex(inputShape, batchIdx, cInput, yInput, xInput);
                    unsigned int stridedInputIdx =
                        dataLayoutIndexed.GetIndex(stridedInput.shape, batchIdx, cInput, yStrided, xStrided);

                    inputDecoder[inputIdx];
                    stridedInput.data[stridedInputIdx] = inputDecoder.Get();
                }
            }
        }
    }

    return stridedInput;
}

TensorData SetUpEmptyPaddedOutput(const TensorShape& outputShape,
                                  const TransposeConvolution2dDescriptor& descriptor,
                                  const DataLayoutIndexed& dataLayoutIndexed)
{
    const unsigned int cIndex = dataLayoutIndexed.GetChannelsIndex();
    const unsigned int hIndex = dataLayoutIndexed.GetHeightIndex();
    const unsigned int wIndex = dataLayoutIndexed.GetWidthIndex();

    const unsigned int batches  = outputShape[0];
    const unsigned int channels = outputShape[cIndex];

    const unsigned int wOutput = outputShape[wIndex];
    const unsigned int hOutput = outputShape[hIndex];

    const unsigned int wPaddedOutput = wOutput + descriptor.m_PadLeft + descriptor.m_PadRight;
    const unsigned int hPaddedOutput = hOutput + descriptor.m_PadTop  + descriptor.m_PadBottom;

    TensorData paddedOutput;
    paddedOutput.data  = std::vector<float>(batches * channels * wPaddedOutput * hPaddedOutput, 0.0f);
    paddedOutput.shape = TensorShape(4);

    paddedOutput.shape[0]      = batches;
    paddedOutput.shape[cIndex] = channels;
    paddedOutput.shape[hIndex] = hPaddedOutput;
    paddedOutput.shape[wIndex] = wPaddedOutput;

    return paddedOutput;
}

void Deconvolve(const TensorData& stridedInput,
                TensorData& paddedOutput,
                const TensorShape& weightsShape,
                Decoder<float>& weightsDecoder,
                const DataLayoutIndexed& dataLayoutIndexed)
{
    const unsigned int cIndex = dataLayoutIndexed.GetChannelsIndex();
    const unsigned int hIndex = dataLayoutIndexed.GetHeightIndex();
    const unsigned int wIndex = dataLayoutIndexed.GetWidthIndex();

    const unsigned int batches  = stridedInput.shape[0];
    const unsigned int channels = stridedInput.shape[cIndex];

    const unsigned int wKernel = weightsShape[wIndex];
    const unsigned int hKernel = weightsShape[hIndex];

    const unsigned int wStridedInput = stridedInput.shape[wIndex];
    const unsigned int hStridedInput = stridedInput.shape[hIndex];

    // loop through all input elements
    for (unsigned int batchIdx = 0u; batchIdx < batches; ++batchIdx)
    {
        for (unsigned int cInput = 0u; cInput < channels; ++cInput)
        {
            for (unsigned int yInput = 0u; yInput < hStridedInput; ++yInput)
            {
                for (unsigned int xInput = 0u; xInput < wStridedInput; ++xInput)
                {
                    // obtain input value
                    unsigned int inputIdx =
                        dataLayoutIndexed.GetIndex(stridedInput.shape, batchIdx, cInput, yInput, xInput);
                    float inputValue = stridedInput.data[inputIdx];

                    // loop through kernel
                    for (unsigned int yKernel = 0u; yKernel < hKernel; ++yKernel)
                    {
                        for (unsigned int xKernel = 0; xKernel < wKernel; ++xKernel)
                        {
                            unsigned int kernelIdx =
                                dataLayoutIndexed.GetIndex(weightsShape, batchIdx, cInput, yKernel, xKernel);

                            weightsDecoder[kernelIdx];
                            float kernelValue = weightsDecoder.Get();

                            unsigned int xOutput = xInput + xKernel;
                            unsigned int yOutput = yInput + yKernel;

                            // compute output increment
                            float outputValue = inputValue * kernelValue;

                            unsigned int outputIdx = dataLayoutIndexed.GetIndex(paddedOutput.shape,
                                                                                batchIdx,
                                                                                cInput,
                                                                                yOutput,
                                                                                xOutput);

                            // set output value
                            paddedOutput.data[outputIdx] += outputValue;
                        }
                    }
                }
            }
        }
    }
}

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

    const unsigned int cIndex = dataLayoutIndexed.GetChannelsIndex();
    const unsigned int hIndex = dataLayoutIndexed.GetHeightIndex();
    const unsigned int wIndex = dataLayoutIndexed.GetWidthIndex();

    const unsigned int numBatches  = inputShape[0];
    const unsigned int numChannels = inputShape[cIndex];

    // set up temporary strided input
    TensorData stridedInput = SetUpStridedInput(inputShape, inputDecoder, descriptor, dataLayoutIndexed);

    // set up temporary (empty) padded output
    TensorData paddedOutput = SetUpEmptyPaddedOutput(outputShape, descriptor, dataLayoutIndexed);

    // run deconvolution (without biases) on strided input to produce padded output
    Deconvolve(stridedInput, paddedOutput, weightsShape, weightsDecoder, dataLayoutIndexed);

    const unsigned int wPaddedOutput = paddedOutput.shape[wIndex];
    const unsigned int hPaddedOutput = paddedOutput.shape[hIndex];

    // remove padding and apply bias (if enabled)
    for (unsigned int batchIdx = 0u; batchIdx < numBatches; ++batchIdx)
    {
        for (unsigned int cOutput = 0u; cOutput < numChannels; ++cOutput)
        {
            // update bias decoder iterator
            if (descriptor.m_BiasEnabled)
            {
                (*biasesDecoder)[cOutput];
            }

            for (unsigned int yPaddedOutput = descriptor.m_PadTop;
                 yPaddedOutput < (hPaddedOutput - descriptor.m_PadBottom);
                 ++yPaddedOutput)
            {
                for (unsigned int xPaddedOutput = descriptor.m_PadLeft;
                     xPaddedOutput < (wPaddedOutput - descriptor.m_PadRight);
                     ++xPaddedOutput)
                {
                    unsigned int xOutput = xPaddedOutput - descriptor.m_PadLeft;
                    unsigned int yOutput = yPaddedOutput - descriptor.m_PadTop;

                    unsigned int outputIdx =
                        dataLayoutIndexed.GetIndex(outputShape, batchIdx, cOutput, yOutput, xOutput);
                    unsigned int paddedOutputIdx =
                        dataLayoutIndexed.GetIndex(paddedOutput.shape, batchIdx, cOutput, yPaddedOutput, xPaddedOutput);

                    // encode (copy) output data
                    outputEncoder[outputIdx];
                    outputEncoder.Set(paddedOutput.data[paddedOutputIdx]);

                    // apply bias (if enabled)
                    if (descriptor.m_BiasEnabled)
                    {
                        outputEncoder.Set(outputEncoder.Get() + biasesDecoder->Get());
                    }
                }
            }
        }
    }
}

} // namespace armnn