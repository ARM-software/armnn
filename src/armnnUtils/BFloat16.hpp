//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ostream>
#include <cmath>
#include <stdint.h>

namespace armnn
{
class BFloat16
{
public:
    BFloat16()
    : value(0)
    {}

    explicit BFloat16(uint16_t v)
    : value(v)
    {}

    explicit BFloat16(float v)
    {
        value = float32ToBFloat16(v).val();
    }

    BFloat16& operator=(float v)
    {
        value = float32ToBFloat16(v).val();
        return *this;
    }

    bool operator==(const BFloat16& r) const
    {
        return value == r.val();
    }

    bool operator==(const float& r) const
    {
        return toFloat32() == r;
    }

    static BFloat16 float32ToBFloat16(const float v)
    {
        if (std::isnan(v))
        {
            return nan();
        }
        else
        {
            // Round value to the nearest even
            // Float32
            // S EEEEEEEE MMMMMMLRMMMMMMMMMMMMMMM
            // BFloat16
            // S EEEEEEEE MMMMMML
            // LSB (L): Least significat bit of BFloat16 (last bit of the Mantissa of BFloat16)
            // R: Rounding bit
            // LSB = 0, R = 0 -> round down
            // LSB = 1, R = 0 -> round down
            // LSB = 0, R = 1, all the rest = 0 -> round down
            // LSB = 1, R = 1 -> round up
            // LSB = 0, R = 1 -> round up
            const uint32_t* u32 = reinterpret_cast<const uint32_t*>(&v);
            uint16_t u16 = static_cast<uint16_t>(*u32 >> 16u);
            // Mark the LSB
            const uint16_t lsb = u16 & 0x0001;
            // Mark the error to be truncate (the rest of 16 bits of FP32)
            const uint16_t error = static_cast<const uint16_t>((*u32 & 0x0000FFFF));
            if ((error > 0x8000 || (error == 0x8000 && lsb == 1)))
            {
                u16++;
            }
            BFloat16 b(u16);
            return b;
        }
    }

    float toFloat32() const
    {
        const uint32_t u32 = static_cast<const uint32_t>(value << 16u);
        const float* f32 = reinterpret_cast<const float*>(&u32);
        return *f32;
    }

    uint16_t val() const
    {
        return value;
    }

    static BFloat16 max()
    {
        uint16_t max = 0x7F7F;
        return BFloat16(max);
    }

    static BFloat16 nan()
    {
        uint16_t nan = 0x7FC0;
        return BFloat16(nan);
    }

    static BFloat16 inf()
    {
        uint16_t infVal = 0x7F80;
        return BFloat16(infVal);
    }

private:
    uint16_t value;
};

inline std::ostream& operator<<(std::ostream& os, const BFloat16& b)
{
    os << b.toFloat32() << "(0x" << std::hex << b.val() << ")";
    return os;
}

} //namespace armnn
