//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/ArmNN.hpp>

#include <ArmnnSchema_generated.h>

namespace armnnSerializer
{

armnnSerializer::ConstTensorData GetFlatBufferConstTensorData(armnn::DataType dataType);

armnnSerializer::DataType GetFlatBufferDataType(armnn::DataType dataType);

armnnSerializer::DataLayout GetFlatBufferDataLayout(armnn::DataLayout dataLayout);

armnnSerializer::PoolingAlgorithm GetFlatBufferPoolingAlgorithm(armnn::PoolingAlgorithm poolingAlgorithm);

armnnSerializer::OutputShapeRounding GetFlatBufferOutputShapeRounding(
    armnn::OutputShapeRounding outputShapeRounding);

armnnSerializer::PaddingMethod GetFlatBufferPaddingMethod(armnn::PaddingMethod paddingMethod);

} // namespace armnnSerializer
