//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ArmNN.hpp>
#include <armnn/TypesUtils.hpp>

#include <initializer_list>
#include <iterator>
#include <vector>

#include <boost/core/ignore_unused.hpp>
#include <boost/numeric/conversion/cast.hpp>

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
        boost::ignore_unused(scale, offset);
        return value;
    }

    static float Dequantize(T value, float scale, int32_t offset)
    {
        boost::ignore_unused(scale, offset);
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
std::vector<T> QuantizedVector(float qScale, int32_t qOffset, FloatIt first, FloatIt last)
{
    std::vector<T> quantized;
    quantized.reserve(boost::numeric_cast<size_t>(std::distance(first, last)));

    for (auto it = first; it != last; ++it)
    {
        auto f = *it;
        T q =SelectiveQuantize<T>(f, qScale, qOffset);
        quantized.push_back(q);
    }

    return quantized;
}

template<typename T>
std::vector<T> QuantizedVector(float qScale, int32_t qOffset, const std::vector<float>& array)
{
    return QuantizedVector<T>(qScale, qOffset, array.begin(), array.end());
}

template<typename T>
std::vector<T> QuantizedVector(float qScale, int32_t qOffset, std::initializer_list<float> array)
{
    return QuantizedVector<T>(qScale, qOffset, array.begin(), array.end());
}
