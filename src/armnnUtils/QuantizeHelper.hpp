//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/TypesUtils.hpp>

#include <BFloat16.hpp>
#include <Half.hpp>

#include <initializer_list>
#include <iterator>
#include <vector>

namespace armnnUtils
{

template<typename T, bool DoQuantize=true>
struct SelectiveQuantizer
{
    static T Quantize(float value, float scale, int32_t offset)
    {
        return armnn::Quantize<T>(value, scale, offset);
    }

    static float Dequantize(T value, float scale, int32_t offset)
    {
        return armnn::Dequantize(value, scale, offset);
    }
};

template<typename T>
struct SelectiveQuantizer<T, false>
{
    static T Quantize(float value, float scale, int32_t offset)
    {
        armnn::IgnoreUnused(scale, offset);
        return value;
    }

    static float Dequantize(T value, float scale, int32_t offset)
    {
        armnn::IgnoreUnused(scale, offset);
        return value;
    }
};

template<>
struct SelectiveQuantizer<armnn::Half, false>
{
    static armnn::Half Quantize(float value, float scale, int32_t offset)
    {
        armnn::IgnoreUnused(scale, offset);
        return armnn::Half(value);
    }

    static float Dequantize(armnn::Half value, float scale, int32_t offset)
    {
        armnn::IgnoreUnused(scale, offset);
        return value;
    }
};

template<>
struct SelectiveQuantizer<armnn::BFloat16, false>
{
    static armnn::BFloat16 Quantize(float value, float scale, int32_t offset)
    {
        armnn::IgnoreUnused(scale, offset);
        return armnn::BFloat16(value);
    }

    static float Dequantize(armnn::BFloat16 value, float scale, int32_t offset)
    {
        armnn::IgnoreUnused(scale, offset);
        return value;
    }
};

template<typename T>
T SelectiveQuantize(float value, float scale, int32_t offset)
{
    return SelectiveQuantizer<T, armnn::IsQuantizedType<T>()>::Quantize(value, scale, offset);
};

template<typename T>
float SelectiveDequantize(T value, float scale, int32_t offset)
{
    return SelectiveQuantizer<T, armnn::IsQuantizedType<T>()>::Dequantize(value, scale, offset);
};

template<typename ItType>
struct IsFloatingPointIterator
{
    static constexpr bool value=std::is_floating_point<typename std::iterator_traits<ItType>::value_type>::value;
};

template <typename T, typename FloatIt,
typename std::enable_if<IsFloatingPointIterator<FloatIt>::value, int>::type=0 // Makes sure fp iterator is valid.
>
std::vector<T> QuantizedVector(FloatIt first, FloatIt last, float qScale, int32_t qOffset)
{
    std::vector<T> quantized;
    quantized.reserve(armnn::numeric_cast<size_t>(std::distance(first, last)));

    for (auto it = first; it != last; ++it)
    {
        auto f = *it;
        T q = SelectiveQuantize<T>(f, qScale, qOffset);
        quantized.push_back(q);
    }

    return quantized;
}

template<typename T>
std::vector<T> QuantizedVector(const std::vector<float>& array, float qScale = 1.f, int32_t qOffset = 0)
{
    return QuantizedVector<T>(array.begin(), array.end(), qScale, qOffset);
}

template<typename T>
std::vector<T> QuantizedVector(std::initializer_list<float> array, float qScale = 1.f, int32_t qOffset = 0)
{
    return QuantizedVector<T>(array.begin(), array.end(), qScale, qOffset);
}

} // namespace armnnUtils
