//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SerializerUtils.hpp"

namespace armnnSerializer
{

using namespace armnn;

armnnSerializer::ConstTensorData GetFlatBufferConstTensorData(armnn::DataType dataType)
{
    switch (dataType)
    {
        case armnn::DataType::Float32:
        case armnn::DataType::Signed32:
            return armnnSerializer::ConstTensorData::ConstTensorData_IntData;
        case armnn::DataType::Float16:
            return armnnSerializer::ConstTensorData::ConstTensorData_ShortData;
        case armnn::DataType::QuantisedAsymm8:
        case armnn::DataType::Boolean:
            return armnnSerializer::ConstTensorData::ConstTensorData_ByteData;
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
        case armnn::DataType::QuantisedAsymm8:
            return armnnSerializer::DataType::DataType_QuantisedAsymm8;
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
        case armnn::DataLayout::NCHW:
        default:
            return armnnSerializer::DataLayout::DataLayout_NCHW;
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

} // namespace armnnSerializer