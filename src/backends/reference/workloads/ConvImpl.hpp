//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefWorkloadUtils.hpp"

#include <armnn/Tensor.hpp>

#include <boost/assert.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <cmath>
#include <limits>

namespace armnn
{

/// Performs multiplication of an integer with a multiplier which is less than one,
/// using quantized integer arithmetic which is consistent with AndroidNN's CPU executor.
struct QuantizedMultiplierSmallerThanOne
{
public:
    /// Constructs a QuantizedMultiplierSmallerThanOne which will multiply by the given multiplier.
    /// This stores the appropriate integer quantities (derived from the given multiplier) for later use.
    /// The implementation of this function is adapted from Android NN's QuantizeMultiplierSmallerThanOne().
    QuantizedMultiplierSmallerThanOne(float multiplier);

    /// The implementation of this function is adapted from Android NN's MultiplyByQuantizedMultiplierSmallerThanOne().
    int32_t operator*(int32_t rhs) const;

private:
    /// The implementation of this function is adapted from gemmlowp's SaturatingRoundingDoublingHighMul().
    static int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b);

    /// The implementation of this function is adapted from gemmlowp's RoundingDivideByPOT().
    static int32_t RoundingDivideByPOT(int32_t x, int exponent);

    int32_t m_Multiplier;
    int32_t m_RightShift;
};

/// An implementation shared by normal and depthwise convolution.
template<typename ConvData, typename InputType, typename BiasType, typename AccumulatorType>
static void ConvImpl(ConvData data,
                     const InputType* inputData,
                     float inputScale,
                     int32_t inputOffset,
                     const InputType* filterData,
                     float filterScale,
                     int32_t filterOffset,
                     const BiasType* biasData,
                     InputType* outputData,
                     float outputScale,
                     int32_t outputOffset,
                     const TensorInfo& filterInfo,
                     bool depthwise = false)
{
    if (data.m_Parameters.m_BiasEnabled && !biasData)
    {
        throw InvalidArgumentException("Bias is enabled but the bias data is invalid");
    }

    const TensorInfo& inputInfo0  = GetTensorInfo(data.m_Inputs[0]);
    const TensorInfo& outputInfo0 = GetTensorInfo(data.m_Outputs[0]);

    const DataLayoutIndexed dataLayoutIndexed(data.m_Parameters.m_DataLayout);
    const unsigned int channelsIndex = dataLayoutIndexed.GetChannelsIndex();
    const unsigned int heightIndex   = dataLayoutIndexed.GetHeightIndex();
    const unsigned int widthIndex    = dataLayoutIndexed.GetWidthIndex();

    unsigned int depthMult      = depthwise ? filterInfo.GetShape()[0] : 1;
    unsigned int channelsInput  = filterInfo.GetShape()[channelsIndex];
    unsigned int channelsOutput = depthwise ? channelsInput * depthMult : filterInfo.GetShape()[0];

    unsigned int batchSize    = outputInfo0.GetShape()[0];
    unsigned int heightOutput = outputInfo0.GetShape()[heightIndex];
    unsigned int widthOutput  = outputInfo0.GetShape()[widthIndex];
    unsigned int heightInput  = inputInfo0.GetShape()[heightIndex];
    unsigned int widthInput   = inputInfo0.GetShape()[widthIndex];

    unsigned int heightFilter = filterInfo.GetShape()[heightIndex];
    unsigned int widthFilter  = filterInfo.GetShape()[widthIndex];

    unsigned int paddingTop = data.m_Parameters.m_PadTop;
    unsigned int paddingLeft = data.m_Parameters.m_PadLeft;
    unsigned int hStride  = data.m_Parameters.m_StrideY;
    unsigned int xStride  = data.m_Parameters.m_StrideX;

    // The world's least efficient convolution.
    for (unsigned int batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        for (unsigned int cOutput = 0; cOutput < channelsOutput; cOutput++)
        {
            for (unsigned int yOutput = 0; yOutput < heightOutput; yOutput++)
            {
                for (unsigned int xOutput = 0; xOutput < widthOutput; xOutput++)
                {
                    // This loop goes over each output element.
                    AccumulatorType sum = AccumulatorType();

                    // For depthwise, each output channel corresponds to exactly one input channel.
                    // For normal, must loop over each input channel.
                    for (unsigned int cInput = 0; cInput < (depthwise ? 1 : channelsInput); cInput++)
                    {
                        unsigned int depthwiseMultiplierIdx = 0;
                        if (depthwise)
                        {
                            cInput = cOutput / depthMult;
                            depthwiseMultiplierIdx = cOutput % depthMult;
                        }

                        for (unsigned int yFilter = 0; yFilter < heightFilter; yFilter++)
                        {
                            for (unsigned int xFilter = 0; xFilter < widthFilter; xFilter++)
                            {
                                // This loop goes over each input element for each output element.

                                unsigned int filterIndex;

                                // Since dimensionality of kernel depends on depthwiseness, so does index.
                                if (depthwise)
                                {
                                    filterIndex = depthwiseMultiplierIdx * widthFilter * heightFilter * channelsInput +
                                                  cInput * widthFilter * heightFilter +
                                                  yFilter * widthFilter +
                                                  xFilter;
                                }
                                else
                                {
                                    filterIndex = cOutput * widthFilter * heightFilter * channelsInput +
                                                  cInput  * widthFilter * heightFilter +
                                                  yFilter * widthFilter +
                                                  xFilter;
                                }
                                AccumulatorType filterValue = filterData[filterIndex] -
                                    boost::numeric_cast<AccumulatorType>(filterOffset);

                                unsigned int yInput = yOutput * hStride + yFilter;
                                unsigned int xInput = xOutput * xStride + xFilter;

                                AccumulatorType inputValue;

                                // Check if we're in the padding.
                                if (yInput < paddingTop || yInput >= heightInput + paddingTop ||
                                    xInput < paddingLeft || xInput >= widthInput + paddingLeft )
                                {
                                    inputValue = AccumulatorType();
                                }
                                else
                                {
                                    inputValue = inputData[batchIdx * widthInput * heightInput * channelsInput +
                                                                      widthInput * heightInput * cInput +
                                                                      widthInput * (yInput - paddingTop) +
                                                                      xInput - paddingLeft] -
                                        boost::numeric_cast<AccumulatorType>(inputOffset);
                                }
                                sum += filterValue * inputValue;
                            }
                        }
                    }

                    if (data.m_Parameters.m_BiasEnabled)
                    {
                        sum += biasData[cOutput];
                    }

                    if (outputScale != 0.0f)
                    {
                        float multiplier = (inputScale * filterScale) / outputScale;
                        // Apply the multiplier to sum, but do so using some quantized arithmetic which is consistent
                        // with the AndroidNN CPU implementation. This should be (roughly) equivalent to:
                        //  sum = std::round(multiplier * sum + outputOffset);
                        sum = boost::numeric_cast<AccumulatorType>(
                                QuantizedMultiplierSmallerThanOne(multiplier) * boost::numeric_cast<int32_t>(sum))
                            + boost::numeric_cast<AccumulatorType>(outputOffset);
                        sum = std::min<AccumulatorType>(std::max<AccumulatorType>(sum, 0), 255);
                    }

                    outputData[batchIdx * widthOutput * heightOutput * channelsOutput +
                                          widthOutput * heightOutput * cOutput +
                                          widthOutput * yOutput +
                                          xOutput] = boost::numeric_cast<InputType>(sum);
                }
            }
        }
    }
}

} //namespace armnn
