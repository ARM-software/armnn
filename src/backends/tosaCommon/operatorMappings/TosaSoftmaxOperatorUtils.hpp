//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gemmlowp/fixedpoint.h>

static void SoftmaxScaling5Bits(double softmaxBeta,
                                double scale,
                                int32_t &input_beta_multiplier,
                                int &input_beta_left_shift)
{
    double max_real_multiplier = (1LL << 31) - 1.0;
    double input_beta_real_multiplier = std::min<double>(softmaxBeta * scale * (1 << (31 - 5)), max_real_multiplier);

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(input_beta_real_multiplier > 1.,
        "CalculateSoftmaxTableValues: QuantizeMultiplierGreaterThanOne must be greater than 1");

    if (input_beta_real_multiplier == 0.)
    {
        input_beta_multiplier = 0;
        input_beta_left_shift = 0;
        return;
    }
    double q = std::frexp(input_beta_real_multiplier, &input_beta_left_shift);
    auto q_fixed = static_cast<int64_t>(std::round(q * (1LL << 31)));

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(q_fixed <= (1LL << 31), "CalculateSoftmaxTableValues: Rounding not valid");

    if (q_fixed == (1LL << 31))
    {
        q_fixed /= 2;
        ++input_beta_left_shift;
    }

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(q_fixed <= std::numeric_limits<int32_t>::max(),
                                        "CalculateSoftmaxTableValues: All results would be zero");

    if (input_beta_left_shift < -31)
    {
        input_beta_left_shift = 0;
        q_fixed = 0;
    }
    input_beta_multiplier = static_cast<int32_t>(q_fixed);
}

static int CalculateInputRadius5Bits(int &input_beta_left_shift)
{
    const double max_input_rescaled = 1.0 * ((1 << 5) - 1) * (1LL << (31 - 5)) /
                                      (static_cast<double>(1LL << input_beta_left_shift));

    return static_cast<int>(std::floor(max_input_rescaled));
}

inline void CalculateSoftmaxTableValues(double softmaxBeta, double scale, std::array<std::vector<int16_t>, 4>& tables)
{
    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(softmaxBeta == 1.0f,
                                        "CalculateSoftmaxTableValues: Beta values other than 1.0 are not supported");

    int32_t input_beta_multiplier = 0;
    int input_beta_left_shift = 0;

    const int kScaledDiffIntegerBits = 5;
    using FixedPointScaledDiff = gemmlowp::FixedPoint<int32_t, kScaledDiffIntegerBits>;
    using gemmlowp::SaturatingRoundingDoublingHighMul;
    using gemmlowp::exp_on_negative_values;

    SoftmaxScaling5Bits(softmaxBeta, scale, input_beta_multiplier, input_beta_left_shift);
    int diff_min = -1 * CalculateInputRadius5Bits(input_beta_left_shift);

    for (int32_t input_diff = -256; input_diff <= 256; input_diff++)
    {
        int32_t output = 0;
        if (input_diff >= diff_min)
        {
            int32_t input_diff_rescaled =
                SaturatingRoundingDoublingHighMul(input_diff * (1 << input_beta_left_shift), input_beta_multiplier);
            const FixedPointScaledDiff input_diff_fixed_point = FixedPointScaledDiff::FromRaw(input_diff_rescaled);
            output = exp_on_negative_values(input_diff_fixed_point).raw();
        }

        // Only copy the 8-bit groups
        int32_t first = (output >> 24) & 0xFF;
        int32_t second = (output >> 16) & 0xFF;
        int32_t third = (output >> 8) & 0xFF;
        int32_t fourth = (output) & 0xFF;
        tables[0].push_back(static_cast<int16_t>(first));
        tables[1].push_back(static_cast<int16_t>(second));
        tables[2].push_back(static_cast<int16_t>(third));
        tables[3].push_back(static_cast<int16_t>(fourth));
    }
}
