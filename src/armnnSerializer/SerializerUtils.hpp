//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <ArmnnSchema_generated.h>

namespace armnnSerializer
{

armnnSerializer::ComparisonOperation GetFlatBufferComparisonOperation(armnn::ComparisonOperation comparisonOperation);

armnnSerializer::ConstTensorData GetFlatBufferConstTensorData(armnn::DataType dataType);

armnnSerializer::DataType GetFlatBufferDataType(armnn::DataType dataType);

armnnSerializer::DataLayout GetFlatBufferDataLayout(armnn::DataLayout dataLayout);

armnnSerializer::UnaryOperation GetFlatBufferUnaryOperation(armnn::UnaryOperation unaryOperation);

armnnSerializer::PoolingAlgorithm GetFlatBufferPoolingAlgorithm(armnn::PoolingAlgorithm poolingAlgorithm);

armnnSerializer::OutputShapeRounding GetFlatBufferOutputShapeRounding(
    armnn::OutputShapeRounding outputShapeRounding);

armnnSerializer::PaddingMethod GetFlatBufferPaddingMethod(armnn::PaddingMethod paddingMethod);

armnnSerializer::PaddingMode GetFlatBufferPaddingMode(armnn::PaddingMode paddingMode);

armnnSerializer::NormalizationAlgorithmChannel GetFlatBufferNormalizationAlgorithmChannel(
    armnn::NormalizationAlgorithmChannel normalizationAlgorithmChannel);

armnnSerializer::NormalizationAlgorithmMethod GetFlatBufferNormalizationAlgorithmMethod(
    armnn::NormalizationAlgorithmMethod normalizationAlgorithmMethod);

armnnSerializer::ResizeMethod GetFlatBufferResizeMethod(armnn::ResizeMethod method);

armnnSerializer::LogicalBinaryOperation GetFlatBufferLogicalBinaryOperation(
    armnn::LogicalBinaryOperation logicalBinaryOperation);

armnnSerializer::ReduceOperation GetFlatBufferReduceOperation(armnn::ReduceOperation reduceOperation);

} // namespace armnnSerializer
