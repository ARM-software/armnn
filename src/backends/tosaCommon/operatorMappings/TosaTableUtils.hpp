//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include <cfloat>
#include <vector>
#include <functional>
#include <cstdint>
#include <cmath>


// Abstract of getTosaConst8bitTable() function from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc
inline std::vector<int16_t> getTosaConst8bitTable(float input_scale,
                                                  int32_t input_zp,
                                                  float output_scale,
                                                  int32_t output_zp,
                                                  std::function<float(float)> func)
{
    // TosaTableAttribute requires int16 vector input. However, TOSA TABLE legalizations are performed using int8.
    std::vector<int16_t> table;
    table.reserve(256);
    float inverse_scale = 1.0f / output_scale;
    for (int32_t i = -128; i < 128; i++)
    {
        float dequantized = input_scale * static_cast<float>(i - input_zp);
        float transformed = func(dequantized);

        float max = (output_scale > 1.0) ? FLT_MAX : (FLT_MAX * output_scale);
        if (transformed >= max)
        {
            table.push_back(INT8_MAX);
            continue;
        }

        int32_t rescaled = static_cast<int32_t>(std::round(transformed * inverse_scale));
        int32_t quantized = static_cast<int32_t>(rescaled + output_zp);
        table.push_back(
            static_cast<int8_t>(std::min(std::max(quantized, -128), 127)));
    }
    return table;
}

// Abstract of getTosaConst16bitTable() function from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc
template <typename FloatT>
inline std::vector<int16_t> getTosaConst16bitTable(float input_scale,
                                                   int32_t input_zp,
                                                   float output_scale,
                                                   int32_t output_zp,
                                                   std::function<FloatT(FloatT)> func)
{
    std::vector<int16_t> table;
    table.reserve(513);

    FloatT input_min =
        input_scale * static_cast<FloatT>(std::numeric_limits<int16_t>::min() - input_zp);
    FloatT input_max =
        input_scale * static_cast<FloatT>(std::numeric_limits<int16_t>::max() - input_zp);
    FloatT output_min =
        output_scale * static_cast<FloatT>(std::numeric_limits<int16_t>::min() - output_zp);
    FloatT output_max =
        output_scale * static_cast<FloatT>(std::numeric_limits<int16_t>::max() - output_zp);

    FloatT step = (input_max - input_min) / 512;
    FloatT half_step = step / 2;
    FloatT output_scaling_inv = 65536 / (output_max - output_min);

    for (int32_t i = 0; i < 512; i++)
    {
        FloatT iFloat = static_cast<FloatT>(i);
        FloatT sample_val =
            std::round(func(input_min + (iFloat * step)) * output_scaling_inv);
        FloatT midpoint_interp_val = std::round(
            ((func(input_min + (iFloat + 1) * step) * output_scaling_inv) +
                std::round(func(input_min + (iFloat * step)) * output_scaling_inv)) /
            2);
        FloatT midpoint_val = std::round(func(input_min + (iFloat * step) + half_step) *
                                            output_scaling_inv);
        FloatT midpoint_err = midpoint_interp_val - midpoint_val;
        FloatT bias = std::round(midpoint_err / 2);

        table.push_back(static_cast<int16_t>(
            std::min<FloatT>(std::max<FloatT>(sample_val - bias, -32768), 32767)));
    }

    FloatT max_val = std::round(func(input_max) * output_scaling_inv);
    table.push_back(static_cast<int16_t>(
        std::min<FloatT>(std::max<FloatT>(max_val, -32768), 32767)));
    return table;
}