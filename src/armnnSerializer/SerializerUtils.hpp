//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/ArmNN.hpp>

#include <Schema_generated.h>

namespace armnnSerializer
{

armnn::armnnSerializer::ConstTensorData GetFlatBufferConstTensorData(armnn::DataType dataType);

armnn::armnnSerializer::DataType GetFlatBufferDataType(armnn::DataType dataType);

armnn::armnnSerializer::DataLayout GetFlatBufferDataLayout(armnn::DataLayout dataLayout);

armnn::armnnSerializer::PoolingAlgorithm GetFlatBufferPoolingAlgorithm(armnn::PoolingAlgorithm poolingAlgorithm);

armnn::armnnSerializer::OutputShapeRounding GetFlatBufferOutputShapeRounding(
    armnn::OutputShapeRounding outputShapeRounding);

armnn::armnnSerializer::PaddingMethod GetFlatBufferPaddingMethod(armnn::PaddingMethod paddingMethod);

} // namespace armnnSerializer