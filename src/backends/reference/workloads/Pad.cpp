//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Pad.hpp"

#include "BaseIterator.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"

#include <armnnUtils/TensorUtils.hpp>

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <cassert>

namespace
{

void FillOutputWithPadValue(armnn::Encoder<float>& outputData,
                            const float padValue,
                            const unsigned int numOutputElements)
{
    for (unsigned int i = 0; i < numOutputElements; ++i)
    {
        outputData[i];
        outputData.Set(padValue);
    }
}

} // anonymous namespace

namespace armnn
{

void Pad(const TensorInfo& inputInfo,
         const TensorInfo& outputInfo,
         const ITensorHandle* inputHandle,
         ITensorHandle* outputHandle,
         const PadQueueDescriptor& data)
{
    auto padList  = data.m_Parameters.m_PadList;
    auto padValue = data.m_Parameters.m_PadValue;

    unsigned int numOutputElements = outputInfo.GetNumElements();

    TensorShape outputShape = outputInfo.GetShape();
    TensorShape inputShape  = inputInfo.GetShape();

    unsigned int numInputDimensions = inputShape.GetNumDimensions();

#ifndef NDEBUG

    unsigned int numOutputDimensions = outputShape.GetNumDimensions();
    assert(numInputDimensions == numOutputDimensions);

#endif

    unsigned int inputBatches  = 0;
    unsigned int inputChannels = 0;
    unsigned int inputHeight   = 0;
    unsigned int inputWidth    = 0;

    unsigned int outputChannels = 0;
    unsigned int outputHeight   = 0;
    unsigned int outputWidth    = 0;

    auto inputData = MakeDecoder<float>(inputInfo, inputHandle->Map());
    auto outData   = MakeEncoder<float>(outputInfo, outputHandle->Map());

    // Fill the output tensor with Pad value first
    if (outputInfo.IsQuantized())
    {
        // For Quantized types Pad Value should not be quantized with scale and offset of the tensor info
        auto temporaryInfo = TensorInfo(outputInfo.GetShape(), outputInfo.GetDataType(), 1.0f, 0);
        auto outputData = MakeEncoder<float>(temporaryInfo, outputHandle->Map());
        FillOutputWithPadValue(*outputData, padValue, numOutputElements);
    }
    else
    {
        FillOutputWithPadValue(*outData, padValue, numOutputElements);
    }

    Decoder<float>& input  = *inputData;
    Encoder<float>& output = *outData;

    switch(numInputDimensions) {

        case 1:
            inputWidth = inputShape[0];
            for (unsigned int w = 0; w < inputWidth ; w++)
            {
                input[w];
                auto inputValue = input.Get();
                auto outputIndex = w + std::get<0>(padList[0]);
                output[outputIndex];
                output.Set(inputValue);
            }

            break;
        case 2  :
            inputHeight = inputShape[0];
            inputWidth  = inputShape[1];
            outputWidth = outputShape[1];

            for (unsigned int h = 0; h < inputHeight; h++)
            {
                for (unsigned int w = 0; w < inputWidth ; w++)
                {
                    input[h * inputWidth + w];
                    auto inputValue  = input.Get();
                    auto outputIndex = (h + std::get<0>(padList[0])) * outputWidth + (w + std::get<0>(padList[1]));
                    output[outputIndex];
                    output.Set(inputValue);
                }
            }

            break;
        case 3  :
            inputChannels = inputShape[0];
            inputHeight   = inputShape[1];
            inputWidth    = inputShape[2];
            outputHeight  = outputShape[1];
            outputWidth   = outputShape[2];

            for (unsigned int c = 0; c < inputChannels; c++)
            {
                for (unsigned int h = 0; h < inputHeight; h++)
                {
                    for (unsigned int w = 0; w < inputWidth ; w++)
                    {
                        input[c * inputHeight * inputWidth + h * inputWidth + w];
                        auto inputValue  = input.Get();
                        auto outputIndex = (c + std::get<0>(padList[0])) * outputHeight * outputWidth
                                           + (h + std::get<0>(padList[1])) * outputWidth
                                           + (w + std::get<0>(padList[2]));
                        output[outputIndex];
                        output.Set(inputValue);
                    }
                }
            }

            break;
        case 4  :
            inputBatches   = inputShape[0];
            inputChannels  = inputShape[1];
            inputHeight    = inputShape[2];
            inputWidth     = inputShape[3];
            outputChannels = outputShape[1];
            outputHeight   = outputShape[2];
            outputWidth    = outputShape[3];

            for (unsigned int b = 0; b < inputBatches; b++)
            {
                for (unsigned int c = 0; c < inputChannels; c++)
                {
                    for (unsigned int h = 0; h < inputHeight; h++)
                    {
                        for (unsigned int w = 0; w < inputWidth ; w++)
                        {
                            input[b * inputChannels * inputHeight * inputWidth
                                      + c * inputHeight * inputWidth
                                      + h * inputWidth
                                      + w];
                            auto inputValue  = input.Get();
                            auto outputIndex = (b + std::get<0>(padList[0]))
                                               * outputChannels * outputHeight * outputWidth
                                               + (c + std::get<0>(padList[1])) * outputHeight * outputWidth
                                               + (h + std::get<0>(padList[2])) * outputWidth
                                               + (w + std::get<0>(padList[3]));
                            output[outputIndex];
                            output.Set(inputValue);
                        }
                    }
                }
            }

            break;
        default :
            break;
    }
}

} //namespace armnn