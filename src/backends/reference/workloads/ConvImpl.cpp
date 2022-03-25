//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConvImpl.hpp"

#include <armnn/utility/Assert.hpp>

#include <cmath>
#include <limits>

namespace armnn
{

QuantizedMultiplierSmallerThanOne::QuantizedMultiplierSmallerThanOne(float multiplier)
{
    ARMNN_ASSERT(multiplier >= 0.0f && multiplier < 1.0f);
    if (multiplier == 0.0f)
    {
        m_Multiplier = 0;
        m_RightShift = 0;
    }
    else
    {
        const double q = std::frexp(multiplier, &m_RightShift);
        m_RightShift = -m_RightShift;
        int64_t qFixed = static_cast<int64_t>(::round(q * (1ll << 31)));
        ARMNN_ASSERT(qFixed <= (1ll << 31));
        if (qFixed == (1ll << 31))
        {
            qFixed /= 2;
            --m_RightShift;
        }
        ARMNN_ASSERT(m_RightShift >= 0);
        ARMNN_ASSERT(qFixed <= std::numeric_limits<int32_t>::max());
        m_Multiplier = static_cast<int32_t>(qFixed);
    }
}

int32_t QuantizedMultiplierSmallerThanOne::operator*(int32_t rhs) const
{
    int32_t x = SaturatingRoundingDoublingHighMul(rhs, m_Multiplier);
    return RoundingDivideByPOT(x, m_RightShift);
}

int32_t QuantizedMultiplierSmallerThanOne::SaturatingRoundingDoublingHighMul(int32_t a, int32_t b)
{
    // Check for overflow.
    if (a == b && a == std::numeric_limits<int32_t>::min())
    {
        return std::numeric_limits<int32_t>::max();
    }
    int64_t a_64(a);
    int64_t b_64(b);
    int64_t ab_64 = a_64 * b_64;
    int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
    int32_t ab_x2_high32 = static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));
    return ab_x2_high32;
}

int32_t QuantizedMultiplierSmallerThanOne::RoundingDivideByPOT(int32_t x, int exponent)
{
    ARMNN_ASSERT(exponent >= 0 && exponent <= 31);
    int32_t mask = (1 << exponent) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

void Convolve(const TensorShape& rInputShape,
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
              unsigned int xStride,
              unsigned int yStride,
              unsigned int xDilation,
              unsigned int yDilation,
              bool depthwise)
{
    if (biasEnabled && !pBiasDecoder)
    {
        throw InvalidArgumentException("Bias is enabled but the bias data is invalid");
    }
    const armnnUtils::DataLayoutIndexed dataLayoutIndexed(dataLayout);

    const unsigned int channelsIndex = dataLayoutIndexed.GetChannelsIndex();
    const unsigned int heightIndex   = dataLayoutIndexed.GetHeightIndex();
    const unsigned int widthIndex    = dataLayoutIndexed.GetWidthIndex();

    // Weights layout:
    // Conv2d:    [O,H,W,I]
    // Depthwise: [1,H,W,O]
    const unsigned int inputChannels   = rInputShape[channelsIndex];
    const unsigned int outputChannels  = rOutputShape[channelsIndex];
    const unsigned int depthMultiplier = depthwise ? outputChannels/inputChannels : 1;

    const unsigned int batchSize    = rOutputShape[0];
    const unsigned int outputHeight = rOutputShape[heightIndex];
    const unsigned int outputWidth  = rOutputShape[widthIndex];
    const unsigned int inputHeight  = rInputShape[heightIndex];
    const unsigned int inputWidth   = rInputShape[widthIndex];

    const unsigned int filterHeight = depthwise ? rFilterShape[1] : rFilterShape[heightIndex];
    const unsigned int filterWidth  = depthwise ? rFilterShape[2] : rFilterShape[widthIndex];

    const std::vector<float> inputVec = rInputDecoder.DecodeTensor(rInputShape);
    const std::vector<float> filterVec = rFilterDecoder.DecodeTensor(rFilterShape, depthwise);

    const TensorShape biasShape{outputChannels};
    const std::vector<float> biasVec = biasEnabled ? pBiasDecoder->DecodeTensor(biasShape) : std::vector<float>();

    for (unsigned int batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        for (unsigned int cOutput = 0; cOutput < outputChannels; cOutput++)
        {
            for (unsigned int yOutput = 0; yOutput < outputHeight; yOutput++)
            {
                for (unsigned int xOutput = 0; xOutput < outputWidth; xOutput++)
                {
                    // This loop goes over each output element.
                    float sum = 0.0f;

                    // For depthwise, each output channel corresponds to exactly one input channel.
                    // For normal, must loop over each input channel.
                    for (unsigned int cInput = 0; cInput < (depthwise ? 1 : inputChannels); cInput++)
                    {
                        for (unsigned int yFilter = 0; yFilter < filterHeight; yFilter++)
                        {
                            for (unsigned int xFilter = 0; xFilter < filterWidth; xFilter++)
                            {
                                // This loop goes over each input element for each output element.
                                unsigned int filterIndex = 0;

                                // Since dimensionality of kernel depends on depthwiseness, so does index.
                                if (depthwise)
                                {
                                    cInput = cOutput / depthMultiplier;
                                    // filterDepth = outputChannels;
                                    filterIndex = xFilter * outputChannels + cOutput +
                                                  yFilter * filterWidth * outputChannels;
                                }
                                else
                                {
                                    // Keep this implementation, as using DataLayoutIndexed::GetIndex causes great
                                    // performance regression.
                                    if (dataLayoutIndexed.GetDataLayout() == DataLayout::NHWC)
                                    {
                                        filterIndex = cOutput * filterHeight * filterWidth * inputChannels +
                                                      yFilter * filterWidth * inputChannels +
                                                      xFilter * inputChannels +
                                                      cInput;
                                    }
                                    else
                                    {
                                        filterIndex = cOutput * filterWidth * filterHeight * inputChannels +
                                                      cInput * filterWidth * filterHeight +
                                                      yFilter * filterWidth +
                                                      xFilter;
                                    }
                                }

                                unsigned int yInput = yOutput * yStride + yFilter * yDilation;
                                unsigned int xInput = xOutput * xStride + xFilter * xDilation;

                                float inputValue;

                                // Check if we're in the padding.
                                if (yInput < paddingTop || yInput >= inputHeight + paddingTop ||
                                    xInput < paddingLeft || xInput >= inputWidth + paddingLeft)
                                {
                                    inputValue = 0.0f;
                                }
                                else
                                {
                                    unsigned int inputIndex = 0;

                                    // Keep this implementation, as using DataLayoutIndexed::GetIndex causes great
                                    // performance regression.
                                    if (dataLayoutIndexed.GetDataLayout() == DataLayout::NHWC)
                                    {
                                        inputIndex = batchIdx * inputHeight * inputWidth * inputChannels +
                                                     (yInput - paddingTop) * inputWidth * inputChannels +
                                                     (xInput - paddingLeft) * inputChannels +
                                                     cInput;
                                    }
                                    else
                                    {
                                        inputIndex = batchIdx * inputWidth * inputHeight * inputChannels +
                                                     inputWidth * inputHeight * cInput +
                                                     inputWidth * (yInput - paddingTop) +
                                                     xInput - paddingLeft;
                                    }
                                    inputValue = inputVec[inputIndex];
                                }

                                sum += filterVec[filterIndex] * inputValue;
                            }
                        }
                    }

                    if (biasEnabled)
                    {
                        sum += biasVec[cOutput];
                    }

                    unsigned int outIdx;
                    if (dataLayoutIndexed.GetDataLayout() == DataLayout::NHWC)
                    {
                        outIdx =  batchIdx * outputHeight * outputWidth * outputChannels +
                                  yOutput * outputWidth * outputChannels +
                                  xOutput * outputChannels +
                                  cOutput;
                    }
                    else
                    {
                        outIdx = batchIdx * outputHeight * outputWidth * outputChannels +
                                 cOutput * outputHeight * outputWidth +
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

} // namespace armnn
