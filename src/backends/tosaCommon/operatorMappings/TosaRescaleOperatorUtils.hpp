//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Exceptions.hpp>

#pragma once

inline void CreateRescaleTosaOperator(const std::string& inputName,
                                      const std::string& outputName,
                                      DType output_type,
                                      const std::vector<int32_t>& shape,
                                      int32_t scale_multiplier,
                                      int32_t scale_shift,
                                      int32_t input_zp,
                                      int32_t output_zp,
                                      bool double_round,
                                      bool scale32,
                                      TosaSerializationOperator** op,
                                      TosaSerializationTensor** tensor)
{
    if (!op)
    {
        throw armnn::Exception("CreateRescaleTosaOperator: nullptr op");
    }

    std::vector<int32_t> multipliers{scale_multiplier};
    std::vector<int32_t> shifts{scale_shift};
    TosaRescaleAttribute attribute(input_zp,
                                   output_zp,
                                   multipliers,
                                   shifts,
                                   scale32,
                                   double_round,
                                   false,  // per_channel
                                   false,  // input_unsigned
                                   false); // output_unsigned

    // op
    *op = new TosaSerializationOperator(Op_RESCALE, Attribute_RescaleAttribute, &attribute, {inputName}, {outputName});
    if (!(*op))
    {
        throw armnn::Exception("CreateRescaleTosaOperator: failed to created operator");
    }
    if (tensor != nullptr)
    {
        // tensor
        *tensor = new TosaSerializationTensor(outputName, shape, output_type, {});
        if (! (*tensor))
        {
            throw armnn::Exception("CreateRescaleTosaOperator: failed to created tensor");
        }
    }
}

inline void CreateRescaleTosaOperator(const std::string& inputName,
                                      const std::string& outputName,
                                      DType output_type,
                                      const std::vector<int32_t>& shape,
                                      double scale,
                                      int32_t input_zp,
                                      int32_t output_zp,
                                      bool double_round,
                                      bool scale32,
                                      TosaSerializationOperator** op,
                                      TosaSerializationTensor** tensor)
{
    //  The code that follows is based on the behaviour specified in
    //  https://www.mlplatform.org/tosa/tosa_spec.html#_precision_scaling

    auto GetScaleParams = [](double scale, double& m, int32_t& n)
    {
        m = 0;
        n = 0;

        double lastErr = 1e06;

        const int32_t numExponents = 62;
        const double start = 1.0;
        const double end = 2.0;

        // Slow iterative approach but running in Reference only
        for (int32_t i = 0; i < numExponents; ++i)
        {
            double exp = 1.0 / (1 << i);
            double currentM = scale / exp;    // Find current m given value = currentM  * exp
            if ((currentM >= start) && (currentM < end))
            {
                double value = currentM * exp;
                double err = std::abs(scale - value);
                if (err < lastErr)
                {
                    // Take the m, n that minimize the error
                    n = i;
                    m = currentM;
                    lastErr = err;
                }
            }
        }
    };

    auto GetMultiplierShiftByScale = [GetScaleParams](bool scale32, double scale, int32_t& multiplier, int32_t& shift)
    {
        double m = 0;
        int32_t n = 0;

        GetScaleParams(scale, m, n);

        multiplier  = (scale32) ? (1 << 30) * static_cast<int32_t>(m) : (1 << 14) * static_cast<int32_t>(m);
        shift       = (scale32) ? (30 + n) : (14 + n);
    };

    int32_t multiplier;
    int32_t shift;
    GetMultiplierShiftByScale(scale32, scale, multiplier, shift);
    CreateRescaleTosaOperator(inputName, outputName, output_type, shape, multiplier, shift,
                              input_zp, output_zp, double_round, scale32, op, tensor);
}

inline void CreateFromInt32RescaleTosaOperator(const std::string& inputName,
                                               const std::string& outputName,
                                                DType output_type,
                                                const std::vector<int32_t>& shape,
                                                double output_scale,
                                                int32_t output_zp,
                                                TosaSerializationOperator** op,
                                                TosaSerializationTensor** tensor)
{
    CreateRescaleTosaOperator(inputName, outputName, output_type, shape,
                              output_scale, 0, output_zp, true, true, op, tensor);
}
