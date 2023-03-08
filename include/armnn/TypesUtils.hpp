//
// Copyright © 2018-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/BackendId.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include <stdint.h>
#include <cmath>
#include <ostream>
#include <set>
#include <type_traits>

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
        case ActivationFunction::Elu:           return "Elu";
        case ActivationFunction::HardSwish:     return "HardSwish";
        default:                                return "Unknown";
    }
}

constexpr char const* GetArgMinMaxFunctionAsCString(ArgMinMaxFunction function)
{
    switch (function)
    {
        case ArgMinMaxFunction::Max:    return "Max";
        case ArgMinMaxFunction::Min:    return "Min";
        default:                        return "Unknown";
    }
}

constexpr char const* GetComparisonOperationAsCString(ComparisonOperation operation)
{
    switch (operation)
    {
        case ComparisonOperation::Equal:          return "Equal";
        case ComparisonOperation::Greater:        return "Greater";
        case ComparisonOperation::GreaterOrEqual: return "GreaterOrEqual";
        case ComparisonOperation::Less:           return "Less";
        case ComparisonOperation::LessOrEqual:    return "LessOrEqual";
        case ComparisonOperation::NotEqual:       return "NotEqual";
        default:                                  return "Unknown";
    }
}

constexpr char const* GetBinaryOperationAsCString(BinaryOperation operation)
{
    switch (operation)
    {
        case BinaryOperation::Add:      return "Add";
        case BinaryOperation::Div:      return "Div";
        case BinaryOperation::Maximum:  return "Maximum";
        case BinaryOperation::Minimum:  return "Minimum";
        case BinaryOperation::Mul:      return "Mul";
        case BinaryOperation::Sub:      return "Sub";
        default:                        return "Unknown";
    }
}

constexpr char const* GetUnaryOperationAsCString(UnaryOperation operation)
{
    switch (operation)
    {
        case UnaryOperation::Abs:        return "Abs";
        case UnaryOperation::Exp:        return "Exp";
        case UnaryOperation::Sqrt:       return "Sqrt";
        case UnaryOperation::Rsqrt:      return "Rsqrt";
        case UnaryOperation::Neg:        return "Neg";
        case UnaryOperation::Log:        return "Log";
        case UnaryOperation::LogicalNot: return "LogicalNot";
        case UnaryOperation::Sin:        return "Sin";
        default:                         return "Unknown";
    }
}

constexpr char const* GetLogicalBinaryOperationAsCString(LogicalBinaryOperation operation)
{
    switch (operation)
    {
        case LogicalBinaryOperation::LogicalAnd: return "LogicalAnd";
        case LogicalBinaryOperation::LogicalOr:  return "LogicalOr";
        default:                                 return "Unknown";
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

constexpr char const* GetPaddingModeAsCString(PaddingMode mode)
{
    switch (mode)
    {
        case PaddingMode::Constant:   return "Exclude";
        case PaddingMode::Symmetric:  return "Symmetric";
        case PaddingMode::Reflect:    return "Reflect";
        default:                      return "Unknown";
    }
}

constexpr char const* GetReduceOperationAsCString(ReduceOperation reduce_operation)
{
    switch (reduce_operation)
    {
        case ReduceOperation::Sum:  return "Sum";
        case ReduceOperation::Max:  return "Max";
        case ReduceOperation::Mean: return "Mean";
        case ReduceOperation::Min:  return "Min";
        case ReduceOperation::Prod: return "Prod";
        default:                    return "Unknown";
    }
}
constexpr unsigned int GetDataTypeSize(DataType dataType)
{
    switch (dataType)
    {
        case DataType::BFloat16:
        case DataType::Float16:               return 2U;
        case DataType::Float32:
        case DataType::Signed32:              return 4U;
        case DataType::Signed64:              return 8U;
        case DataType::QAsymmU8:              return 1U;
        case DataType::QAsymmS8:              return 1U;
        case DataType::QSymmS8:               return 1U;
        case DataType::QSymmS16:              return 2U;
        case DataType::Boolean:               return 1U;
        default:                              return 0U;
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
        case DataType::Float16:               return "Float16";
        case DataType::Float32:               return "Float32";
        case DataType::Signed64:              return "Signed64";
        case DataType::QAsymmU8:              return "QAsymmU8";
        case DataType::QAsymmS8:              return "QAsymmS8";
        case DataType::QSymmS8:               return "QSymmS8";
        case DataType::QSymmS16:              return "QSymm16";
        case DataType::Signed32:              return "Signed32";
        case DataType::Boolean:               return "Boolean";
        case DataType::BFloat16:              return "BFloat16";

        default:
            return "Unknown";
    }
}

constexpr const char* GetDataLayoutName(DataLayout dataLayout)
{
    switch (dataLayout)
    {
        case DataLayout::NCHW:  return "NCHW";
        case DataLayout::NHWC:  return "NHWC";
        case DataLayout::NDHWC: return "NDHWC";
        case DataLayout::NCDHW: return "NCDHW";
        default:                return "Unknown";
    }
}

constexpr const char* GetNormalizationAlgorithmChannelAsCString(NormalizationAlgorithmChannel channel)
{
    switch (channel)
    {
        case NormalizationAlgorithmChannel::Across: return "Across";
        case NormalizationAlgorithmChannel::Within: return "Within";
        default:                                    return "Unknown";
    }
}

constexpr const char* GetNormalizationAlgorithmMethodAsCString(NormalizationAlgorithmMethod method)
{
    switch (method)
    {
        case NormalizationAlgorithmMethod::LocalBrightness: return "LocalBrightness";
        case NormalizationAlgorithmMethod::LocalContrast:   return "LocalContrast";
        default:                                            return "Unknown";
    }
}

constexpr const char* GetResizeMethodAsCString(ResizeMethod method)
{
    switch (method)
    {
        case ResizeMethod::Bilinear:        return "Bilinear";
        case ResizeMethod::NearestNeighbor: return "NearestNeighbour";
        default:                            return "Unknown";
    }
}

constexpr const char* GetMemBlockStrategyTypeName(MemBlockStrategyType memBlockStrategyType)
{
    switch (memBlockStrategyType)
    {
        case MemBlockStrategyType::SingleAxisPacking: return "SingleAxisPacking";
        case MemBlockStrategyType::MultiAxisPacking:  return "MultiAxisPacking";
        default:                                      return "Unknown";
    }
}

template<typename T>
struct IsHalfType
    : std::integral_constant<bool, std::is_floating_point<T>::value && sizeof(T) == 2>
{};

template<typename T>
constexpr bool IsQuantizedType()
{
    return std::is_integral<T>::value;
}

constexpr bool IsQuantized8BitType(DataType dataType)
{
    return dataType == DataType::QAsymmU8        ||
           dataType == DataType::QAsymmS8        ||
           dataType == DataType::QSymmS8;
}

constexpr bool IsQuantizedType(DataType dataType)
{
    return dataType == DataType::QSymmS16 || IsQuantized8BitType(dataType);
}

inline std::ostream& operator<<(std::ostream& os, Status stat)
{
    os << GetStatusAsCString(stat);
    return os;
}


inline std::ostream& operator<<(std::ostream& os, const armnn::TensorShape& shape)
{
    os << "[";
    if (shape.GetDimensionality() != Dimensionality::NotSpecified)
    {
        for (uint32_t i = 0; i < shape.GetNumDimensions(); ++i)
        {
            if (i != 0)
            {
                os << ",";
            }
            if (shape.GetDimensionSpecificity(i))
            {
                os << shape[i];
            }
            else
            {
                os << "?";
            }
        }
    }
    else
    {
        os << "Dimensionality Not Specified";
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
QuantizedType Quantize(float value, float scale, int32_t offset);

/// Dequantize an 8-bit data type into a floating point data type.
/// @param value - The value to dequantize.
/// @param scale - The scale (must be non-zero).
/// @param offset - The offset.
/// @return - The dequantized value calculated as (value-offset)*scale.
///
template <typename QuantizedType>
float Dequantize(QuantizedType value, float scale, int32_t offset);

inline void VerifyTensorInfoDataType(const armnn::TensorInfo & info, armnn::DataType dataType)
{
    if (info.GetDataType() != dataType)
    {
        std::stringstream ss;
        ss << "Unexpected datatype:" << armnn::GetDataTypeName(info.GetDataType())
           << " for tensor:" << info.GetShape()
           << ". The type expected to be: " << armnn::GetDataTypeName(dataType);
        throw armnn::Exception(ss.str());
    }
}

} //namespace armnn
