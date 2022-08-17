//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include <tosa_generated.h>

using namespace armnn;
using namespace tosa;

// Function to return Tosa datatype from input ArmNN datatype.
DType ArmNNToDType(const DataType& type)
{
    switch (type)
    {
        case DataType::Float16:
        case DataType::Float32:
        case DataType::BFloat16:
            return DType_FLOAT;
        case DataType::QAsymmU8:
            return DType_UINT8;
        case DataType::QSymmS8:
        case DataType::QAsymmS8:
            return DType_INT8;
        case DataType::QSymmS16:
            return DType_INT16;
        case DataType::Signed32:
            return DType_INT32;
        case DataType::Signed64:
            // No signed 64, only DType_INT48.
            return DType_UNKNOWN;
        case DataType::Boolean:
            return DType_BOOL;
        default:
            return DType_UNKNOWN;
    }
}

// Function to return Tosa tensor shape from input ArmNN tensor shape.
std::vector<int32_t> GetTosaTensorShape(const TensorShape& shape)
{
    std::vector<int32_t> returnShape;
    for (u_int32_t i = 0; i < shape.GetNumDimensions(); i++)
    {
        returnShape.push_back(static_cast<int32_t>(shape[i]));
    }
    return returnShape;
}

// Function to return unique int as a string to ensure uniqueness between all input, output and block names.
static int uniqueTosaMappingID = 0;
std::string GetUniqueTosaMappingID()
{
    return std::to_string(++uniqueTosaMappingID);
}
