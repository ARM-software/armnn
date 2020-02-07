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

struct QAsymmU8QuantizationScheme : IQuantizationScheme
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

        // To avoid dividing by zero when quantizing a zero filled tensor
        if (min == 0.0 && max == 0.0)
        {
            max = 1.0;
        }

        // Assumes quantization range [0-highest]
        double scale = (max-min) / highest;
        double offset = -min / scale;

        // Clamp offset [0-highest]
        offset = std::max(0.0, std::min(highest, offset));

        return std::make_pair(static_cast<float>(scale), static_cast<int>(std::round(offset)));
    }

    int NumBits() const override { return 8; }

    DataType GetDataType() const override { return DataType::QAsymmU8; }
};

struct QAsymmS8QuantizationScheme : IQuantizationScheme
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

        // To avoid dividing by zero when quantizing a zero filled tensor
        if (min == 0.0 && max == 0.0)
        {
            max = 1.0;
        }

        // Assumes quantization range [0-255]
        double scale = (max-min) / highest ;
        double offset = - min / scale;

        //Clamp 0 to Highest
        offset = std::max(0.0, std::min(highest, offset));

        //-128 on offset to cast to signed range
        return std::make_pair(static_cast<float>(scale), static_cast<int>(std::round(offset)-128));
    }

    int NumBits() const override { return 8; }

    DataType GetDataType() const override { return DataType::QAsymmS8; }
};

struct QSymmS8QuantizationScheme : IQuantizationScheme
{
    OffsetScalePair ComputeScheme(double min, double max) const override
    {
        if (min > max)
        {
            throw InvalidArgumentException("min > max will result in invalid quantization.");
        }

        // To avoid dividing by zero when quantizing a zero filled tensor
        if (min == 0.0 && max == 0.0)
        {
            max = 1.0;
        }

        double highest = (1 << (NumBits()-1)) - 1; // (numbits-1) accounts for the sign bit

        double extent = std::max(std::abs(min), std::abs(max));
        double scale = extent / highest;

        return std::make_pair(static_cast<float>(scale), 0);
    }

    int NumBits() const override { return 8; }

    DataType GetDataType() const override { return DataType::QSymmS8; }
};

struct QSymm16QuantizationScheme : IQuantizationScheme
{
    OffsetScalePair ComputeScheme(double min, double max) const override
    {
        if (min > max)
        {
            throw InvalidArgumentException("min > max will result in invalid quantization.");
        }

        // To avoid dividing by zero when quantizing a zero filled tensor
        if (min == 0.0 && max == 0.0)
        {
            max = 1.0;
        }

        double highest = (1 << (NumBits()-1)) - 1; // (numbits-1) accounts for the sign bit

        double extent = std::max(std::abs(min), std::abs(max));
        double scale = extent / highest;

        return std::make_pair(static_cast<float>(scale), 0);

    }

    int NumBits() const override { return 16; }

    DataType GetDataType() const override { return DataType::QSymmS16; }
};

} // namespace armnn
