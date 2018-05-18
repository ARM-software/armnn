//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "Types.hpp"
#include "Tensor.hpp"
#include <cmath>
#include <ostream>
#include <boost/assert.hpp>
#include <boost/numeric/conversion/cast.hpp>

namespace armnn
{

constexpr char const* GetStatusAsCString(Status compute)
{
    switch (compute)
    {
        case armnn::Status::Success: return "Status::Success";
        case armnn::Status::Failure: return "Status::Failure";
        default:                     return "Unknown";
    }
}

constexpr char const* GetComputeDeviceAsCString(Compute compute)
{
    switch (compute)
    {
        case armnn::Compute::CpuRef: return "CpuRef";
        case armnn::Compute::CpuAcc: return "CpuAcc";
        case armnn::Compute::GpuAcc: return "GpuAcc";
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
        case DataType::Signed32:
        case DataType::Float32:   return 4U;
        case DataType::QuantisedAsymm8: return 1U;
        default:                  return 0U;
    }
}

template <int N>
constexpr bool StrEqual(const char* strA, const char (&strB)[N])
{
    bool isEqual = true;
    for (int i = 0; isEqual && (i < N); ++i)
    {
        isEqual = (strA[i] == strB[i]);
    }
    return isEqual;
}

constexpr Compute ParseComputeDevice(const char* str)
{
    if (StrEqual(str, "CpuAcc"))
    {
        return armnn::Compute::CpuAcc;
    }
    else if (StrEqual(str, "CpuRef"))
    {
        return armnn::Compute::CpuRef;
    }
    else if (StrEqual(str, "GpuAcc"))
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
        case DataType::Float32:   return "Float32";
        case DataType::QuantisedAsymm8: return "Unsigned8";
        case DataType::Signed32:  return "Signed32";
        default:                  return "Unknown";
    }
}

template <typename T>
constexpr DataType GetDataType();

template <>
constexpr DataType GetDataType<float>()
{
    return DataType::Float32;
}

template <>
constexpr DataType GetDataType<uint8_t>()
{
    return DataType::QuantisedAsymm8;
}

template <>
constexpr DataType GetDataType<int32_t>()
{
    return DataType::Signed32;
}

template<typename T>
constexpr bool IsQuantizedType()
{
    return std::is_integral<T>::value;
}


template<DataType DT>
struct ResolveTypeImpl;

template<>
struct ResolveTypeImpl<DataType::QuantisedAsymm8>
{
    using Type = uint8_t;
};

template<>
struct ResolveTypeImpl<DataType::Float32>
{
    using Type = float;
};

template<DataType DT>
using ResolveType = typename ResolveTypeImpl<DT>::Type;


inline std::ostream& operator<<(std::ostream& os, Status stat)
{
    os << GetStatusAsCString(stat);
    return os;
}

inline std::ostream& operator<<(std::ostream& os, Compute compute)
{
    os << GetComputeDeviceAsCString(compute);
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

/// Quantize a floating point data type into an 8-bit data type
/// @param value The value to quantize
/// @param scale The scale (must be non-zero)
/// @param offset The offset
/// @return The quantized value calculated as round(value/scale)+offset
///
template<typename QuantizedType>
inline QuantizedType Quantize(float value, float scale, int32_t offset)
{
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

/// Dequantize an 8-bit data type into a floating point data type
/// @param value The value to dequantize
/// @param scale The scale (must be non-zero)
/// @param offset The offset
/// @return The dequantized value calculated as (value-offset)*scale
///
template <typename QuantizedType>
inline float Dequantize(QuantizedType value, float scale, int32_t offset)
{
    static_assert(IsQuantizedType<QuantizedType>(), "Not an integer type.");
    BOOST_ASSERT(scale != 0.f);
    float dequantized = boost::numeric_cast<float>(value - offset) * scale;
    return dequantized;
}

} //namespace armnn
