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
inline DType ArmNNToDType(const DataType& type)
{
    switch (type)
    {
        case DataType::Float16:
        case DataType::BFloat16:
            return DType_FP16;
        case DataType::Float32:
            return DType_FP32;
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
inline std::vector<int32_t> GetTosaTensorShape(const TensorShape& shape)
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
inline std::string GetUniqueTosaMappingID()
{
    return std::to_string(++uniqueTosaMappingID);
}

// Function to return Tosa Op as string.
inline std::string TosaOpToString(Op tosaOp)
{
    switch (tosaOp)
    {
        case Op_ADD:
            return "Op_ADD";
        case Op_AVG_POOL2D:
            return "Op_AVG_POOL2D";
        case Op_MAX_POOL2D:
            return "Op_MAX_POOL2D";
        case Op_PAD:
            return "Op_PAD";
        case Op_UNKNOWN:
            return "Op_UNKNOWN";
        case Op_ARGMAX:
            return "Op_ARGMAX";
        case Op_CONV2D:
            return "Op_CONV2D";
        case Op_CONV3D:
            return "Op_CONV3D";
        case Op_DEPTHWISE_CONV2D:
            return "Op_DEPTHWISE_CONV2D";
        case Op_FULLY_CONNECTED:
            return "Op_FULLY_CONNECTED";
        case Op_MATMUL:
            return "Op_MATMUL";
        case Op_TRANSPOSE_CONV2D:
            return "Op_TRANSPOSE_CONV2D";
        case Op_CLAMP:
            return "Op_CLAMP";
        case Op_RESERVED:
            return "Op_RESERVED";
        case Op_SIGMOID:
            return "Op_SIGMOID";
        case Op_TANH:
            return "Op_TANH";
        case Op_ARITHMETIC_RIGHT_SHIFT:
            return "Op_ARITHMETIC_RIGHT_SHIFT";
        case Op_BITWISE_AND:
            return "Op_BITWISE_AND";
        case Op_BITWISE_OR:
            return "Op_BITWISE_OR";
        case Op_BITWISE_XOR:
            return "Op_BITWISE_XOR";
        case Op_INTDIV:
            return "Op_INTDIV";
        case Op_LOGICAL_AND:
            return "Op_LOGICAL_AND";
        case Op_LOGICAL_LEFT_SHIFT:
            return "Op_LOGICAL_LEFT_SHIFT";
        case Op_LOGICAL_RIGHT_SHIFT:
            return "Op_LOGICAL_RIGHT_SHIFT";
        case Op_LOGICAL_OR:
            return "Op_LOGICAL_OR";
        case Op_LOGICAL_XOR:
            return "Op_LOGICAL_XOR";
        case Op_MAXIMUM:
            return "Op_MAXIMUM";
        case Op_MINIMUM:
            return "Op_MINIMUM";
        case Op_MUL:
            return "Op_MUL";
        case Op_POW:
            return "Op_POW";
        case Op_SUB:
            return "Op_SUB";
        case Op_TABLE:
            return "Op_TABLE";
        case Op_ABS:
            return "Op_ABS";
        case Op_BITWISE_NOT:
            return "Op_BITWISE_NOT";
        case Op_CEIL:
            return "Op_CEIL";
        case Op_CLZ:
            return "Op_CLZ";
        case Op_EXP:
            return "Op_EXP";
        case Op_FLOOR:
            return "Op_FLOOR";
        case Op_LOG:
            return "Op_LOG";
        case Op_LOGICAL_NOT:
            return "Op_LOGICAL_NOT";
        case Op_NEGATE:
            return "Op_NEGATE";
        case Op_RECIPROCAL:
            return "Op_RECIPROCAL";
        case Op_RSQRT:
            return "Op_RSQRT";
        case Op_SELECT:
            return "Op_SELECT";
        case Op_EQUAL:
            return "Op_EQUAL";
        case Op_GREATER:
            return "Op_GREATER";
        case Op_GREATER_EQUAL:
            return "Op_GREATER_EQUAL";
        case Op_REDUCE_ANY:
            return "Op_REDUCE_ANY";
        case Op_REDUCE_ALL:
            return "Op_REDUCE_ALL";
        case Op_REDUCE_MAX:
            return "Op_REDUCE_MAX";
        case Op_REDUCE_MIN:
            return "Op_REDUCE_MIN";
        case Op_REDUCE_PRODUCT:
            return "Op_REDUCE_PRODUCT";
        case Op_REDUCE_SUM:
            return "Op_REDUCE_SUM";
        case Op_CONCAT:
            return "Op_CONCAT";
        case Op_RESHAPE:
            return "Op_RESHAPE";
        case Op_REVERSE:
            return "Op_REVERSE";
        case Op_SLICE:
            return "Op_SLICE";
        case Op_TILE:
            return "Op_TILE";
        case Op_TRANSPOSE:
            return "Op_TRANSPOSE";
        case Op_GATHER:
            return "Op_GATHER";
        case Op_SCATTER:
            return "Op_SCATTER";
        case Op_RESIZE:
            return "Op_RESIZE";
        case Op_CAST:
            return "Op_CAST";
        case Op_RESCALE:
            return "Op_RESCALE";
        case Op_CONST:
            return "Op_CONST";
        case Op_IDENTITY:
            return "Op_IDENTITY";
        case Op_CUSTOM:
            return "Op_CUSTOM";
        case Op_COND_IF:
            return "Op_COND_IF";
        case Op_WHILE_LOOP:
            return "Op_WHILE_LOOP";
    }
    return "";
}
