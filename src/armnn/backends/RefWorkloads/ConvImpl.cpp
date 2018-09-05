//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConvImpl.hpp"

#include <boost/assert.hpp>

#include <cmath>
#include <limits>

namespace armnn
{

QuantizedMultiplierSmallerThanOne::QuantizedMultiplierSmallerThanOne(float multiplier)
{
    BOOST_ASSERT(multiplier >= 0.0f && multiplier < 1.0f);
    if (multiplier == 0.0f)
    {
        m_Multiplier = 0;
        m_RightShift = 0;
    }
    else
    {
        const double q = std::frexp(multiplier, &m_RightShift);
        m_RightShift = -m_RightShift;
        int64_t qFixed = static_cast<int64_t>(std::round(q * (1ll << 31)));
        BOOST_ASSERT(qFixed <= (1ll << 31));
        if (qFixed == (1ll << 31))
        {
            qFixed /= 2;
            --m_RightShift;
        }
        BOOST_ASSERT(m_RightShift >= 0);
        BOOST_ASSERT(qFixed <= std::numeric_limits<int32_t>::max());
        m_Multiplier = static_cast<int32_t>(qFixed);
    }
}

int32_t QuantizedMultiplierSmallerThanOne::operator*(int32_t rhs) const
{
    int32_t x = SaturatingRoundingDoublingHighMul(rhs, m_Multiplier);
    return RoundingDivideByPOT(x, m_RightShift);
}

int32_t QuantizedMultiplierSmallerThanOne::SaturatingRoundingDoublingHighMul(int32_t a, int32_t b)
{
    // Check for overflow.
    if (a == b && a == std::numeric_limits<int32_t>::min())
    {
        return std::numeric_limits<int32_t>::max();
    }
    int64_t a_64(a);
    int64_t b_64(b);
    int64_t ab_64 = a_64 * b_64;
    int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
    int32_t ab_x2_high32 = static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));
    return ab_x2_high32;
}

int32_t QuantizedMultiplierSmallerThanOne::RoundingDivideByPOT(int32_t x, int exponent)
{
    BOOST_ASSERT(exponent >= 0 && exponent <= 31);
    int32_t mask = (1 << exponent) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

} //namespace armnn
