//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SerializerUtils.hpp"

namespace armnnSerializer
{

using namespace armnn;
namespace serializer = armnn::armnnSerializer;

serializer::ConstTensorData GetFlatBufferConstTensorData(DataType dataType)
{
    switch (dataType)
    {
        case DataType::Float32:
        case DataType::Signed32:
            return serializer::ConstTensorData::ConstTensorData_IntData;
        case DataType::Float16:
            return serializer::ConstTensorData::ConstTensorData_ShortData;
        case DataType::QuantisedAsymm8:
        case DataType::Boolean:
            return serializer::ConstTensorData::ConstTensorData_ByteData;
        default:
            return serializer::ConstTensorData::ConstTensorData_NONE;
    }
}

serializer::DataType GetFlatBufferDataType(DataType dataType)
{
    switch (dataType)
    {
        case DataType::Float32:
            return serializer::DataType::DataType_Float32;
        case DataType::Float16:
            return serializer::DataType::DataType_Float16;
        case DataType::Signed32:
            return serializer::DataType::DataType_Signed32;
        case DataType::QuantisedAsymm8:
            return serializer::DataType::DataType_QuantisedAsymm8;
        case DataType::Boolean:
            return serializer::DataType::DataType_Boolean;
        default:
            return serializer::DataType::DataType_Float16;
    }
}

serializer::DataLayout GetFlatBufferDataLayout(DataLayout dataLayout)
{
    switch (dataLayout)
    {
        case DataLayout::NHWC:
            return serializer::DataLayout::DataLayout_NHWC;
        case DataLayout::NCHW:
        default:
            return serializer::DataLayout::DataLayout_NCHW;
    }
}

serializer::PoolingAlgorithm GetFlatBufferPoolingAlgorithm(PoolingAlgorithm poolingAlgorithm)
{
    switch (poolingAlgorithm)
    {
        case PoolingAlgorithm::Average:
            return serializer::PoolingAlgorithm::PoolingAlgorithm_Average;
        case PoolingAlgorithm::L2:
            return serializer::PoolingAlgorithm::PoolingAlgorithm_L2;
        case PoolingAlgorithm::Max:
        default:
            return serializer::PoolingAlgorithm::PoolingAlgorithm_Max;
    }
}

serializer::OutputShapeRounding GetFlatBufferOutputShapeRounding(OutputShapeRounding outputShapeRounding)
{
    switch (outputShapeRounding)
    {
        case OutputShapeRounding::Ceiling:
            return serializer::OutputShapeRounding::OutputShapeRounding_Ceiling;
        case OutputShapeRounding::Floor:
        default:
            return serializer::OutputShapeRounding::OutputShapeRounding_Floor;
    }
}

serializer::PaddingMethod GetFlatBufferPaddingMethod(PaddingMethod paddingMethod)
{
    switch (paddingMethod)
    {
        case PaddingMethod::IgnoreValue:
            return serializer::PaddingMethod::PaddingMethod_IgnoreValue;
        case PaddingMethod::Exclude:
        default:
            return serializer::PaddingMethod::PaddingMethod_Exclude;
    }
}

} // namespace armnnSerializer