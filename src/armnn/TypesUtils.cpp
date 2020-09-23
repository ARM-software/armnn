//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <armnn/TypesUtils.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>

namespace
{
/// Workaround for std:isnan() not being implemented correctly for integral types in MSVC.
/// https://stackoverflow.com/a/56356405
/// @{
template <typename T, typename std::enable_if<std::is_integral<T>::value, T>::type* = nullptr>
inline int IsNan(T x)
{
    // The spec defines integral types to be handled as if they were casted to doubles.
    return std::isnan(static_cast<double>(x));
}

template <typename T, typename std::enable_if<!std::is_integral<T>::value, T>::type * = nullptr>
inline int IsNan(T x)
{
    return std::isnan(x);
}
/// @}
}    // namespace std

template<typename QuantizedType>
QuantizedType armnn::Quantize(float value, float scale, int32_t offset)
{
    static_assert(IsQuantizedType<QuantizedType>(), "Not an integer type.");
    constexpr QuantizedType max = std::numeric_limits<QuantizedType>::max();
    constexpr QuantizedType min = std::numeric_limits<QuantizedType>::lowest();
    ARMNN_ASSERT(scale != 0.f);
    ARMNN_ASSERT(!std::isnan(value));

    float clampedValue = std::min(std::max(static_cast<float>(round(value/scale) + offset), static_cast<float>(min)),
                                  static_cast<float>(max));
    auto quantizedBits = static_cast<QuantizedType>(clampedValue);

    return quantizedBits;
}

template <typename QuantizedType>
float armnn::Dequantize(QuantizedType value, float scale, int32_t offset)
{
    static_assert(IsQuantizedType<QuantizedType>(), "Not an integer type.");
    ARMNN_ASSERT(scale != 0.f);
    ARMNN_ASSERT(!IsNan(value));
    return (armnn::numeric_cast<float>(value - offset)) * scale;
}

/// Explicit specialization of Quantize for int8_t
template
int8_t armnn::Quantize<int8_t>(float value, float scale, int32_t offset);

/// Explicit specialization of Quantize for uint8_t
template
uint8_t armnn::Quantize<uint8_t>(float value, float scale, int32_t offset);

/// Explicit specialization of Quantize for int16_t
template
int16_t armnn::Quantize<int16_t>(float value, float scale, int32_t offset);

/// Explicit specialization of Quantize for int32_t
template
int32_t armnn::Quantize<int32_t>(float value, float scale, int32_t offset);

/// Explicit specialization of Dequantize for int8_t
template
float armnn::Dequantize<int8_t>(int8_t value, float scale, int32_t offset);

/// Explicit specialization of Dequantize for uint8_t
template
float armnn::Dequantize<uint8_t>(uint8_t value, float scale, int32_t offset);

/// Explicit specialization of Dequantize for int16_t
template
float armnn::Dequantize<int16_t>(int16_t value, float scale, int32_t offset);

/// Explicit specialization of Dequantize for int32_t
template
float armnn::Dequantize<int32_t>(int32_t value, float scale, int32_t offset);