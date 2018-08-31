//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Types.hpp"
#include "Tensor.hpp"
#include <cmath>
#include <ostream>
#include <boost/assert.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <set>

namespace armnn
{

constexpr char const* GetStatusAsCString(Status status)
{
    switch (status)
    {
        case armnn::Status::Success: return "Status::Success";
        case armnn::Status::Failure: return "Status::Failure";
        default:                     return "Unknown";
    }
}

constexpr char const* GetActivationFunctionAsCString(ActivationFunction activation)
{
    switch (activation)
    {
        case ActivationFunction::Sigmoid:       return "Sigmoid";
        case ActivationFunction::TanH:          return "TanH";
        case ActivationFunction::Linear:        return "Linear";
        case ActivationFunction::ReLu:          return "ReLu";
        case ActivationFunction::BoundedReLu:   return "BoundedReLu";
        case ActivationFunction::SoftReLu:      return "SoftReLu";
        case ActivationFunction::LeakyReLu:     return "LeakyReLu";
        case ActivationFunction::Abs:           return "Abs";
        case ActivationFunction::Sqrt:          return "Sqrt";
        case ActivationFunction::Square:        return "Square";
        default:                                return "Unknown";
    }
}

constexpr char const* GetPoolingAlgorithmAsCString(PoolingAlgorithm pooling)
{
    switch (pooling)
    {
        case PoolingAlgorithm::Average:  return "Average";
        case PoolingAlgorithm::Max:      return "Max";
        case PoolingAlgorithm::L2:       return "L2";
        default:                         return "Unknown";
    }
}

constexpr char const* GetOutputShapeRoundingAsCString(OutputShapeRounding rounding)
{
    switch (rounding)
    {
        case OutputShapeRounding::Ceiling:  return "Ceiling";
        case OutputShapeRounding::Floor:    return "Floor";
        default:                            return "Unknown";
    }
}


constexpr char const* GetPaddingMethodAsCString(PaddingMethod method)
{
    switch (method)
    {
        case PaddingMethod::Exclude:       return "Exclude";
        case PaddingMethod::IgnoreValue:   return "IgnoreValue";
        default:                           return "Unknown";
    }
}

constexpr unsigned int GetDataTypeSize(DataType dataType)
{
    switch (dataType)
    {
        case DataType::Float16:     return 2U;
        case DataType::Float32:
        case DataType::Signed32:   return 4U;
        case DataType::QuantisedAsymm8: return 1U;
        default:                  return 0U;
    }
}

template <unsigned N>
constexpr bool StrEqual(const char* strA, const char (&strB)[N])
{
    bool isEqual = true;
    for (unsigned i = 0; isEqual && (i < N); ++i)
    {
        isEqual = (strA[i] == strB[i]);
    }
    return isEqual;
}

/// Deprecated function that will be removed together with
/// the Compute enum
constexpr armnn::Compute ParseComputeDevice(const char* str)
{
    if (armnn::StrEqual(str, "CpuAcc"))
    {
        return armnn::Compute::CpuAcc;
    }
    else if (armnn::StrEqual(str, "CpuRef"))
    {
        return armnn::Compute::CpuRef;
    }
    else if (armnn::StrEqual(str, "GpuAcc"))
    {
        return armnn::Compute::GpuAcc;
    }
    else
    {
        return armnn::Compute::Undefined;
    }
}

constexpr const char* GetDataTypeName(DataType dataType)
{
    switch (dataType)
    {
        case DataType::Float16:         return "Float16";
        case DataType::Float32:         return "Float32";
        case DataType::QuantisedAsymm8: return "Unsigned8";
        case DataType::Signed32:        return "Signed32";

        default:
            return "Unknown";
    }
}


template<typename T>
struct IsHalfType
    : std::integral_constant<bool, std::is_floating_point<T>::value && sizeof(T) == 2>
{};

template<typename T, typename U=T>
struct GetDataTypeImpl;

template<typename T>
struct GetDataTypeImpl<T, typename std::enable_if_t<IsHalfType<T>::value, T>>
{
    static constexpr DataType Value = DataType::Float16;
};

template<>
struct GetDataTypeImpl<float>
{
    static constexpr DataType Value = DataType::Float32;
};

template<>
struct GetDataTypeImpl<uint8_t>
{
    static constexpr DataType Value = DataType::QuantisedAsymm8;
};

template<>
struct GetDataTypeImpl<int32_t>
{
    static constexpr DataType Value = DataType::Signed32;
};

template <typename T>
constexpr DataType GetDataType()
{
    return GetDataTypeImpl<T>::Value;
}

template<typename T>
constexpr bool IsQuantizedType()
{
    return std::is_integral<T>::value;
}

inline std::ostream& operator<<(std::ostream& os, Status stat)
{
    os << GetStatusAsCString(stat);
    return os;
}


inline std::ostream & operator<<(std::ostream & os, const armnn::TensorShape & shape)
{
    os << "[";
    for (uint32_t i=0; i<shape.GetNumDimensions(); ++i)
    {
        if (i!=0)
        {
            os << ",";
        }
        os << shape[i];
    }
    os << "]";
    return os;
}

/// Quantize a floating point data type into an 8-bit data type.
/// @param value - The value to quantize.
/// @param scale - The scale (must be non-zero).
/// @param offset - The offset.
/// @return - The quantized value calculated as round(value/scale)+offset.
///
template<typename QuantizedType>
inline QuantizedType Quantize(float value, float scale, int32_t offset)
{
    // TODO : check we act sensibly for Inf, NaN and -Inf
    //        see IVGCVSW-1849
    static_assert(IsQuantizedType<QuantizedType>(), "Not an integer type.");
    constexpr QuantizedType max = std::numeric_limits<QuantizedType>::max();
    constexpr QuantizedType min = std::numeric_limits<QuantizedType>::lowest();
    BOOST_ASSERT(scale != 0.f);
    int quantized = boost::numeric_cast<int>(round(value / scale)) + offset;
    QuantizedType quantizedBits = quantized <= min
                                  ? min
                                  : quantized >= max
                                    ? max
                                    : static_cast<QuantizedType>(quantized);
    return quantizedBits;
}

/// Dequantize an 8-bit data type into a floating point data type.
/// @param value - The value to dequantize.
/// @param scale - The scale (must be non-zero).
/// @param offset - The offset.
/// @return - The dequantized value calculated as (value-offset)*scale.
///
template <typename QuantizedType>
inline float Dequantize(QuantizedType value, float scale, int32_t offset)
{
    static_assert(IsQuantizedType<QuantizedType>(), "Not an integer type.");
    BOOST_ASSERT(scale != 0.f);
    float dequantized = boost::numeric_cast<float>(value - offset) * scale;
    return dequantized;
}

template <typename DataType>
void VerifyTensorInfoDataType(const armnn::TensorInfo & info)
{
    auto expectedType = armnn::GetDataType<DataType>();
    if (info.GetDataType() != expectedType)
    {
        std::stringstream ss;
        ss << "Unexpected datatype:" << armnn::GetDataTypeName(info.GetDataType())
            << " for tensor:" << info.GetShape()
            << ". The type expected to be: " << armnn::GetDataTypeName(expectedType);
        throw armnn::Exception(ss.str());
    }
}

} //namespace armnn
