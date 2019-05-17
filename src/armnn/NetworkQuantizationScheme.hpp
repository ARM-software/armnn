//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>

#include <cmath>
#include <algorithm>

namespace armnn
{

using OffsetScalePair = std::pair<float, int>;

struct IQuantizationScheme
{
    virtual OffsetScalePair ComputeScheme(double min, double max) const = 0;

    virtual int NumBits() const = 0;

    virtual DataType GetDataType() const = 0;

    virtual ~IQuantizationScheme() {}
};

struct QAsymm8QuantizationScheme : IQuantizationScheme
{
    OffsetScalePair ComputeScheme(double min, double max) const override
    {
        if (min > max)
        {
            throw InvalidArgumentException("min > max will result in invalid quantization.");
        }

        double highest = (1 << NumBits()) - 1;

        min = std::min(0.0, min); // min <= 0.0
        max = std::max(0.0, max); // max >= 0.0

        // Assumes quantization range [0-highest]
        double scale = (max-min) / highest;
        double offset = -min / scale;

        // Clamp offset [0-highest]
        offset = std::max(0.0, std::min(highest, offset));

        return std::make_pair(static_cast<float>(scale), static_cast<int>(std::round(offset)));
    }

    int NumBits() const override { return 8; }

    DataType GetDataType() const override { return DataType::QuantisedAsymm8; }
};

struct QSymm16QuantizationScheme : IQuantizationScheme
{
    OffsetScalePair ComputeScheme(double min, double max) const override
    {
        if (min > max)
        {
            throw InvalidArgumentException("min > max will result in invalid quantization.");
        }

        double highest = (1 << (NumBits()-1)) - 1; // (numbits-1) accounts for the sign bit

        double extent = std::max(std::abs(min), std::abs(max));
        double scale = extent / highest;

        return std::make_pair(static_cast<float>(scale), 0);
    }

    int NumBits() const override { return 16; }

    DataType GetDataType() const override { return DataType::QuantisedSymm16; }
};

} // namespace armnn
