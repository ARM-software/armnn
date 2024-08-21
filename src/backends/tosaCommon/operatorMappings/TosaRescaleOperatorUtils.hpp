//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Exceptions.hpp>

#pragma once

inline void CreateRescaleTosaOperator(const std::string& inputName,
                                      const std::string& outputName,
                                      const std::vector<int32_t>& multipliers,
                                      const std::vector<int32_t>& shifts,
                                      int32_t input_zp,
                                      int32_t output_zp,
                                      bool double_round,
                                      bool scale32,
                                      bool per_channel,
                                      TosaSerializationOperator** op)
{
    if (!op)
    {
        throw armnn::Exception("CreateRescaleTosaOperator: nullptr op");
    }

    TosaRescaleAttribute attribute(input_zp,
                                   output_zp,
                                   multipliers,
                                   shifts,
                                   scale32,
                                   double_round,
                                   per_channel,
                                   false,  // input_unsigned
                                   false); // output_unsigned

    // op
    *op = new TosaSerializationOperator(Op_RESCALE, Attribute_RescaleAttribute, &attribute, {inputName}, {outputName});
    if (!(*op))
    {
        throw armnn::Exception("CreateRescaleTosaOperator: failed to created operator");
    }
}

inline void CreateRescaleTosaOperator(const std::string& inputName,
                                      const std::string& outputName,
                                      int32_t scale_multiplier,
                                      int32_t scale_shift,
                                      int32_t input_zp,
                                      int32_t output_zp,
                                      bool double_round,
                                      bool scale32,
                                      bool per_channel,
                                      TosaSerializationOperator** op)
{
    const std::vector<int32_t> multipliers{scale_multiplier};
    const std::vector<int32_t> shifts{scale_shift};
    CreateRescaleTosaOperator(inputName, outputName, multipliers, shifts,
                              input_zp, output_zp, double_round, scale32, per_channel, op);
}

/// The following is taken from mlir/lib/Dialect/Tosa/Utils/QuantUtils.cpp in the LLVM project
/// From a scale value, generates multiplier and shift values where
/// mantissa is in [-1.0,-0.5] or [0.5, 1.0] such that
/// multiplier = mantissa*2^shift for 32-bit scaling.
inline void ComputeMultiplierAndShiftTosaScale32(double scale,
                                                 int32_t &multiplier,
                                                 int32_t &shift)
{
    const double mantissa = std::frexp(scale, &shift);
    auto shiftedM = std::round(mantissa * (int64_t(1) << 31));

    // Can't be greater than 1.0.
    if (!(shiftedM <= (int64_t(1) << 31)))
    {
        throw armnn::Exception("Shifted mantissa exceeds 32 signed bits");
    }

    if (shiftedM == (int64_t(1) << 31))
    {
        shiftedM /= 2;
        shift++;
    }

    // TOSA expects right shift to be positive, and embed (1 << 31) into right
    // shift bits.
    shift = (-shift) + 31;

    if (!(shiftedM <= std::numeric_limits<int32_t>::max()))
    {
        throw armnn::Exception("Shifted mantissa exceeds 32-bit signed output type");
    }

    multiplier = static_cast<int32_t>(shiftedM);

    // Shifting tops out at 62 bits. Right shift to make 62 bits the max.
    // The limit of 62 on shift allows the shift to be decomposed as
    // two right shifts of 31.
    if (shift > 62)
    {
        // Shifting the multiplier by more than 32-bits is unnecessary.
        multiplier = multiplier >> std::min<int32_t>(31, shift - 62);
        shift = 62;
    }
}

/// The following is taken from mlir/lib/Dialect/Tosa/Utils/QuantUtils.cpp in the LLVM project
/// From a scale value, generates multiplier and shift values where
/// mantissa is in [-1.0,-0.5] or [0.5, 1.0] such that
/// multiplier = mantissa*2^shift for 16-bit scaling.
inline void ComputeMultiplierAndShiftTosaScale16(double scale,
                                                 int32_t &multiplier,
                                                 int32_t &shift)
{
    const double mantissa = std::frexp(scale, &shift);
    auto shiftedM = std::round(mantissa * (int64_t(1) << 15));

    // Can't be greater than 1.0.
    if (!(shiftedM <= (int64_t(1) << 15)))
    {
        throw armnn::Exception("Shifted mantissa exceeds 16 signed bits");
    }

    if (shiftedM == (int64_t(1) << 15))
    {
        shiftedM /= 2;
        shift++;
    }

    // TOSA expects right shift to be positive and embed (1 << 15) into right
    // shift bits.
    shift = (-shift) + 15;

    if (!(shiftedM <= std::numeric_limits<int32_t>::max()))
    {
        throw armnn::Exception("Shifted mantissa exceeds 32-bit signed output type");
    }

    multiplier = static_cast<int32_t>(shiftedM);

    // Shifting tops out at 62 bits. Right shift to make 62 bits the max.
    // The limit of 62 on shift allows the shift to be decomposed as
    // two right shifts of 31.
    if (shift > 62)
    {
        // Shifting the multiplier by more than 31-bits is unnecessary.
        multiplier = multiplier >> std::min<int32_t>(31, shift - 62);
        shift = 62;
    }
}

inline void CreateRescaleTosaOperator(const std::string& inputName,
                                      const std::string& outputName,
                                      double scale,
                                      int32_t input_zp,
                                      int32_t output_zp,
                                      bool double_round,
                                      bool scale32,
                                      TosaSerializationOperator** op)
{
    int32_t multiplier;
    int32_t shift;

    if (scale32)
    {
        ComputeMultiplierAndShiftTosaScale32(scale, multiplier, shift);
    }
    else
    {
        ComputeMultiplierAndShiftTosaScale16(scale, multiplier, shift);
    }

    CreateRescaleTosaOperator(inputName, outputName, multiplier, shift,
                              input_zp, output_zp, double_round, scale32, false, op);
}

inline void CreateRescaleTosaOperatorPerChannel(const std::string& inputName,
                                                const std::string& outputName,
                                                int32_t input_zp,
                                                int32_t output_zp,
                                                bool double_round,
                                                bool scale32,
                                                double input_scale,
                                                double output_scale,
                                                const std::vector<float>& weight_scales,
                                                TosaSerializationOperator** op)
{
    std::vector<int32_t> op_tensor_multipliers;
    std::vector<int32_t> op_tensor_shifts;
    op_tensor_multipliers.reserve(weight_scales.size());
    op_tensor_shifts.reserve(weight_scales.size());

    for (const float& weight_scale : weight_scales)
    {
        double op_tensor_scale = (input_scale * weight_scale) / output_scale;
        int32_t multiplier;
        int32_t shift;

        if (scale32)
        {
            ComputeMultiplierAndShiftTosaScale32(op_tensor_scale, multiplier, shift);
        }
        else
        {
            ComputeMultiplierAndShiftTosaScale16(op_tensor_scale, multiplier, shift);
        }

        op_tensor_multipliers.push_back(multiplier);
        op_tensor_shifts.push_back(shift);
    }

    bool per_channel = weight_scales.size() == 1 ? false : true;
    CreateRescaleTosaOperator(inputName, outputName, op_tensor_multipliers, op_tensor_shifts,
                              input_zp, output_zp, double_round, scale32, per_channel, op);
}

inline void CreateFromInt32RescaleTosaOperator(const std::string& inputName,
                                               const std::string& outputName,
                                               double output_scale,
                                               int32_t output_zp,
                                               TosaSerializationOperator** op)
{
    CreateRescaleTosaOperator(inputName, outputName, output_scale,
                              0, output_zp, true, true, op);
}
