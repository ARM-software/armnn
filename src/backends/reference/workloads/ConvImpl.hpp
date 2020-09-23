//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "RefWorkloadUtils.hpp"
#include "TensorBufferArrayView.hpp"
#include "BaseIterator.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"

#include <armnn/Tensor.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

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
              bool depthwise = false);
} //namespace armnn
