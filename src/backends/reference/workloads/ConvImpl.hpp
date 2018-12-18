//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefWorkloadUtils.hpp"
#include "TensorBufferArrayView.hpp"

#include <armnn/Tensor.hpp>

#include <DataLayoutIndexed.hpp>

#include <boost/assert.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <DataLayoutIndexed.hpp>

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
                     float outputScale,
                     int32_t outputOffset,
                     const TensorInfo& filterInfo,
                     bool depthwise = false)
{
    if (data.m_Parameters.m_BiasEnabled && !biasData)
    {
        throw InvalidArgumentException("Bias is enabled but the bias data is invalid");
    }

    const TensorInfo& inputInfo  = GetTensorInfo(data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(data.m_Outputs[0]);

    TensorBufferArrayView<InputType> output(outputInfo.GetShape(),
                                            GetOutputTensorData<InputType>(0, data),
                                            data.m_Parameters.m_DataLayout);

    const armnnUtils::DataLayoutIndexed dataLayoutIndexed(data.m_Parameters.m_DataLayout);

    const unsigned int channelsIndex = dataLayoutIndexed.GetChannelsIndex();
    const unsigned int heightIndex   = dataLayoutIndexed.GetHeightIndex();
    const unsigned int widthIndex    = dataLayoutIndexed.GetWidthIndex();

    unsigned int depthMultiplier = depthwise ? filterInfo.GetShape()[0] : 1;
    unsigned int inputChannels   = depthwise ? filterInfo.GetShape()[1] : filterInfo.GetShape()[channelsIndex];
    unsigned int outputChannels  = depthwise ? inputChannels * depthMultiplier : filterInfo.GetShape()[0];

    unsigned int batchSize    = outputInfo.GetShape()[0];
    unsigned int outputHeight = outputInfo.GetShape()[heightIndex];
    unsigned int outputWidth  = outputInfo.GetShape()[widthIndex];
    unsigned int inputHeight  = inputInfo.GetShape()[heightIndex];
    unsigned int inputWidth   = inputInfo.GetShape()[widthIndex];

    unsigned int filterHeight = depthwise ? filterInfo.GetShape()[2] : filterInfo.GetShape()[heightIndex];
    unsigned int filterWidth  = depthwise ? filterInfo.GetShape()[3] : filterInfo.GetShape()[widthIndex];

    unsigned int paddingTop  = data.m_Parameters.m_PadTop;
    unsigned int paddingLeft = data.m_Parameters.m_PadLeft;
    unsigned int xStride     = data.m_Parameters.m_StrideX;
    unsigned int yStride     = data.m_Parameters.m_StrideY;

    // The world's least efficient convolution.
    for (unsigned int batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        for (unsigned int cOutput = 0; cOutput < outputChannels; cOutput++)
        {
            for (unsigned int yOutput = 0; yOutput < outputHeight; yOutput++)
            {
                for (unsigned int xOutput = 0; xOutput < outputWidth; xOutput++)
                {
                    // This loop goes over each output element.
                    AccumulatorType sum = AccumulatorType();

                    // For depthwise, each output channel corresponds to exactly one input channel.
                    // For normal, must loop over each input channel.
                    for (unsigned int cInput = 0; cInput < (depthwise ? 1 : inputChannels); cInput++)
                    {
                        unsigned int depthwiseMultiplierIdx = 0;
                        if (depthwise)
                        {
                            cInput = cOutput / depthMultiplier;
                            depthwiseMultiplierIdx = cOutput % depthMultiplier;
                        }

                        for (unsigned int yFilter = 0; yFilter < filterHeight; yFilter++)
                        {
                            for (unsigned int xFilter = 0; xFilter < filterWidth; xFilter++)
                            {
                                // This loop goes over each input element for each output element.

                                unsigned int filterIndex = 0;

                                // Since dimensionality of kernel depends on depthwiseness, so does index.
                                if (depthwise)
                                {
                                    filterIndex = depthwiseMultiplierIdx * filterWidth * filterHeight * inputChannels +
                                                  cInput * filterWidth * filterHeight +
                                                  yFilter * filterWidth +
                                                  xFilter;
                                }
                                else
                                {
                                    if (data.m_Parameters.m_DataLayout == DataLayout::NHWC)
                                    {
                                        filterIndex = cOutput * filterHeight * filterWidth * inputChannels +
                                                      yFilter * filterWidth * inputChannels +
                                                      xFilter * inputChannels +
                                                      cInput;
                                    }
                                    else
                                    {
                                        filterIndex = cOutput * filterWidth * filterHeight * inputChannels +
                                                      cInput  * filterWidth * filterHeight +
                                                      yFilter * filterWidth +
                                                      xFilter;
                                    }
                                }

                                AccumulatorType filterValue = filterData[filterIndex] -
                                    boost::numeric_cast<AccumulatorType>(filterOffset);

                                unsigned int yInput = yOutput * yStride + yFilter;
                                unsigned int xInput = xOutput * xStride + xFilter;

                                AccumulatorType inputValue;

                                // Check if we're in the padding.
                                if (yInput < paddingTop || yInput >= inputHeight + paddingTop ||
                                    xInput < paddingLeft || xInput >= inputWidth + paddingLeft )
                                {
                                    inputValue = AccumulatorType();
                                }
                                else
                                {
                                    unsigned int inputIndex;

                                    if (data.m_Parameters.m_DataLayout == DataLayout::NHWC)
                                    {
                                        inputIndex = batchIdx * inputHeight * inputWidth  * inputChannels +
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

                                    inputValue = inputData[inputIndex] -
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

                    output.Get(batchIdx, cOutput, yOutput, xOutput) = boost::numeric_cast<InputType>(sum);
                }
            }
        }
    }
}

} //namespace armnn
