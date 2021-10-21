//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SerializerUtils.hpp"

namespace armnnSerializer
{

armnnSerializer::ComparisonOperation GetFlatBufferComparisonOperation(armnn::ComparisonOperation comparisonOperation)
{
    switch (comparisonOperation)
    {
        case armnn::ComparisonOperation::Equal:
            return armnnSerializer::ComparisonOperation::ComparisonOperation_Equal;
        case armnn::ComparisonOperation::Greater:
            return armnnSerializer::ComparisonOperation::ComparisonOperation_Greater;
        case armnn::ComparisonOperation::GreaterOrEqual:
            return armnnSerializer::ComparisonOperation::ComparisonOperation_GreaterOrEqual;
        case armnn::ComparisonOperation::Less:
            return armnnSerializer::ComparisonOperation::ComparisonOperation_Less;
        case armnn::ComparisonOperation::LessOrEqual:
            return armnnSerializer::ComparisonOperation::ComparisonOperation_LessOrEqual;
        case armnn::ComparisonOperation::NotEqual:
        default:
            return armnnSerializer::ComparisonOperation::ComparisonOperation_NotEqual;
    }
}

armnnSerializer::LogicalBinaryOperation GetFlatBufferLogicalBinaryOperation(
    armnn::LogicalBinaryOperation logicalBinaryOperation)
{
    switch (logicalBinaryOperation)
    {
        case armnn::LogicalBinaryOperation::LogicalAnd:
            return armnnSerializer::LogicalBinaryOperation::LogicalBinaryOperation_LogicalAnd;
        case armnn::LogicalBinaryOperation::LogicalOr:
            return armnnSerializer::LogicalBinaryOperation::LogicalBinaryOperation_LogicalOr;
        default:
            throw armnn::InvalidArgumentException("Logical Binary operation unknown");
    }
}

armnnSerializer::ConstTensorData GetFlatBufferConstTensorData(armnn::DataType dataType)
{
    switch (dataType)
    {
        case armnn::DataType::Float32:
        case armnn::DataType::Signed32:
            return armnnSerializer::ConstTensorData::ConstTensorData_IntData;
        case armnn::DataType::Float16:
        case armnn::DataType::QSymmS16:
            return armnnSerializer::ConstTensorData::ConstTensorData_ShortData;
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QSymmS8:
        case armnn::DataType::Boolean:
            return armnnSerializer::ConstTensorData::ConstTensorData_ByteData;
        case armnn::DataType::Signed64:
            return armnnSerializer::ConstTensorData::ConstTensorData_LongData;
        default:
            return armnnSerializer::ConstTensorData::ConstTensorData_NONE;
    }
}

armnnSerializer::DataType GetFlatBufferDataType(armnn::DataType dataType)
{
    switch (dataType)
    {
        case armnn::DataType::Float32:
            return armnnSerializer::DataType::DataType_Float32;
        case armnn::DataType::Float16:
            return armnnSerializer::DataType::DataType_Float16;
        case armnn::DataType::Signed32:
            return armnnSerializer::DataType::DataType_Signed32;
        case armnn::DataType::Signed64:
            return armnnSerializer::DataType::DataType_Signed64;
        case armnn::DataType::QSymmS16:
            return armnnSerializer::DataType::DataType_QSymmS16;
        case armnn::DataType::QAsymmS8:
            return armnnSerializer::DataType::DataType_QAsymmS8;
        case armnn::DataType::QAsymmU8:
            return armnnSerializer::DataType::DataType_QAsymmU8;
        case armnn::DataType::QSymmS8:
            return armnnSerializer::DataType::DataType_QSymmS8;
        case armnn::DataType::Boolean:
            return armnnSerializer::DataType::DataType_Boolean;
        default:
            return armnnSerializer::DataType::DataType_Float16;
    }
}

armnnSerializer::DataLayout GetFlatBufferDataLayout(armnn::DataLayout dataLayout)
{
    switch (dataLayout)
    {
        case armnn::DataLayout::NHWC:
            return armnnSerializer::DataLayout::DataLayout_NHWC;
        case armnn::DataLayout::NDHWC:
            return armnnSerializer::DataLayout::DataLayout_NDHWC;
        case armnn::DataLayout::NCDHW:
            return armnnSerializer::DataLayout::DataLayout_NCDHW;
        case armnn::DataLayout::NCHW:
        default:
            return armnnSerializer::DataLayout::DataLayout_NCHW;
    }
}

armnnSerializer::UnaryOperation GetFlatBufferUnaryOperation(armnn::UnaryOperation comparisonOperation)
{
    switch (comparisonOperation)
    {
        case armnn::UnaryOperation::Abs:
            return armnnSerializer::UnaryOperation::UnaryOperation_Abs;
        case armnn::UnaryOperation::Rsqrt:
            return armnnSerializer::UnaryOperation::UnaryOperation_Rsqrt;
        case armnn::UnaryOperation::Sqrt:
            return armnnSerializer::UnaryOperation::UnaryOperation_Sqrt;
        case armnn::UnaryOperation::Exp:
            return armnnSerializer::UnaryOperation::UnaryOperation_Exp;
        case armnn::UnaryOperation::Neg:
            return armnnSerializer::UnaryOperation::UnaryOperation_Neg;
        case armnn::UnaryOperation::LogicalNot:
            return armnnSerializer::UnaryOperation::UnaryOperation_LogicalNot;
        case armnn::UnaryOperation::Log:
            return armnnSerializer::UnaryOperation::UnaryOperation_Log;
        case armnn::UnaryOperation::Sin:
            return armnnSerializer::UnaryOperation::UnaryOperation_Sin;
        default:
            throw armnn::InvalidArgumentException("Unary operation unknown");
    }
}

armnnSerializer::PoolingAlgorithm GetFlatBufferPoolingAlgorithm(armnn::PoolingAlgorithm poolingAlgorithm)
{
    switch (poolingAlgorithm)
    {
        case armnn::PoolingAlgorithm::Average:
            return armnnSerializer::PoolingAlgorithm::PoolingAlgorithm_Average;
        case armnn::PoolingAlgorithm::L2:
            return armnnSerializer::PoolingAlgorithm::PoolingAlgorithm_L2;
        case armnn::PoolingAlgorithm::Max:
        default:
            return armnnSerializer::PoolingAlgorithm::PoolingAlgorithm_Max;
    }
}

armnnSerializer::OutputShapeRounding GetFlatBufferOutputShapeRounding(armnn::OutputShapeRounding outputShapeRounding)
{
    switch (outputShapeRounding)
    {
        case armnn::OutputShapeRounding::Ceiling:
            return armnnSerializer::OutputShapeRounding::OutputShapeRounding_Ceiling;
        case armnn::OutputShapeRounding::Floor:
        default:
            return armnnSerializer::OutputShapeRounding::OutputShapeRounding_Floor;
    }
}

armnnSerializer::PaddingMethod GetFlatBufferPaddingMethod(armnn::PaddingMethod paddingMethod)
{
    switch (paddingMethod)
    {
        case armnn::PaddingMethod::IgnoreValue:
            return armnnSerializer::PaddingMethod::PaddingMethod_IgnoreValue;
        case armnn::PaddingMethod::Exclude:
        default:
            return armnnSerializer::PaddingMethod::PaddingMethod_Exclude;
    }
}

armnnSerializer::PaddingMode GetFlatBufferPaddingMode(armnn::PaddingMode paddingMode)
{
    switch (paddingMode)
    {
        case armnn::PaddingMode::Reflect:
            return armnnSerializer::PaddingMode::PaddingMode_Reflect;
        case armnn::PaddingMode::Symmetric:
            return armnnSerializer::PaddingMode::PaddingMode_Symmetric;
        default:
            return armnnSerializer::PaddingMode::PaddingMode_Constant;
    }
}

armnnSerializer::NormalizationAlgorithmChannel GetFlatBufferNormalizationAlgorithmChannel(
    armnn::NormalizationAlgorithmChannel normalizationAlgorithmChannel)
{
    switch (normalizationAlgorithmChannel)
    {
        case armnn::NormalizationAlgorithmChannel::Across:
            return armnnSerializer::NormalizationAlgorithmChannel::NormalizationAlgorithmChannel_Across;
        case armnn::NormalizationAlgorithmChannel::Within:
            return armnnSerializer::NormalizationAlgorithmChannel::NormalizationAlgorithmChannel_Within;
        default:
            return armnnSerializer::NormalizationAlgorithmChannel::NormalizationAlgorithmChannel_Across;
    }
}

armnnSerializer::NormalizationAlgorithmMethod GetFlatBufferNormalizationAlgorithmMethod(
    armnn::NormalizationAlgorithmMethod normalizationAlgorithmMethod)
{
    switch (normalizationAlgorithmMethod)
    {
        case armnn::NormalizationAlgorithmMethod::LocalBrightness:
            return armnnSerializer::NormalizationAlgorithmMethod::NormalizationAlgorithmMethod_LocalBrightness;
        case armnn::NormalizationAlgorithmMethod::LocalContrast:
            return armnnSerializer::NormalizationAlgorithmMethod::NormalizationAlgorithmMethod_LocalContrast;
        default:
            return armnnSerializer::NormalizationAlgorithmMethod::NormalizationAlgorithmMethod_LocalBrightness;
    }
}

armnnSerializer::ResizeMethod GetFlatBufferResizeMethod(armnn::ResizeMethod method)
{
    switch (method)
    {
        case armnn::ResizeMethod::NearestNeighbor:
            return armnnSerializer::ResizeMethod_NearestNeighbor;
        case armnn::ResizeMethod::Bilinear:
            return armnnSerializer::ResizeMethod_Bilinear;
        default:
            return armnnSerializer::ResizeMethod_NearestNeighbor;
    }
}

armnnSerializer::ReduceOperation GetFlatBufferReduceOperation(armnn::ReduceOperation reduceOperation)
{
    switch (reduceOperation)
    {
        case armnn::ReduceOperation::Sum:
            return armnnSerializer::ReduceOperation::ReduceOperation_Sum;
        case armnn::ReduceOperation::Max:
            return armnnSerializer::ReduceOperation::ReduceOperation_Max;
        case armnn::ReduceOperation::Mean:
            return armnnSerializer::ReduceOperation::ReduceOperation_Mean;
        case armnn::ReduceOperation::Min:
            return armnnSerializer::ReduceOperation::ReduceOperation_Min;
        case armnn::ReduceOperation::Prod:
            return armnnSerializer::ReduceOperation::ReduceOperation_Prod;
        default:
            return armnnSerializer::ReduceOperation::ReduceOperation_Sum;
    }
}

} // namespace armnnSerializer
