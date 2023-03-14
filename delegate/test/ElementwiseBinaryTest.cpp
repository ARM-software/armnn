//
// Copyright © 2020-2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ElementwiseBinaryTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <schema_generated.h>
#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void AddFP32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 2, 2, 2, 3 };
    std::vector<int32_t> input1Shape { 2, 2, 2, 3 };
    std::vector<int32_t> expectedOutputShape { 2, 2, 2, 3 };

    std::vector<float> input0Values =
    {
        0.0f, 2.0f, 1.0f,
        0.2f, 1.0f, 2.0f,

        1.0f, 2.0f, 1.0f,
        0.2f, 1.0f, 2.0f,

        0.0f, 2.0f, 1.0f,
        4.2f, 1.0f, 2.0f,

        0.0f, 0.0f, 1.0f,
        0.2f, 1.0f, 2.0f,
    };

    std::vector<float> input1Values =
    {
        1.0f, 2.0f,  1.0f,
        0.0f, 1.0f,  2.0f,

        1.0f, 2.0f, -2.0f,
        0.2f, 1.0f,  2.0f,

        0.0f, 2.0f,  1.0f,
        4.2f, 0.0f, -3.0f,

        0.0f, 0.0f,  1.0f,
        0.7f, 1.0f,  5.0f,
    };

    std::vector<float> expectedOutputValues =
    {
        1.0f, 4.0f,  2.0f,
        0.2f, 2.0f,  4.0f,

        2.0f, 4.0f, -1.0f,
        0.4f, 2.0f,  4.0f,

        0.0f, 4.0f,  2.0f,
        8.4f, 1.0f, -1.0f,

        0.0f, 0.0f,  2.0f,
        0.9f, 2.0f,  7.0f,
    };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_ADD,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void AddBroadcastTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 3, 2, 1 };
    std::vector<int32_t> input1Shape { 1, 1, 2, 3 };
    std::vector<int32_t> expectedOutputShape { 1, 3, 2, 3 };

    std::vector<float> input0Values
    {
        0.0f,
        1.0f,

        2.0f,
        3.0f,

        4.0f,
        5.0f,
    };
    std::vector<float> input1Values
    {
        0.5f, 1.5f, 2.5f,
        3.5f, 4.5f, 5.5f,
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        0.5f, 1.5f, 2.5f,
        4.5f, 5.5f, 6.5f,

        2.5f, 3.5f, 4.5f,
        6.5f, 7.5f, 8.5f,

        4.5f, 5.5f, 6.5f,
        8.5f, 9.5f, 10.5f,
    };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_ADD,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void AddConstInputTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 3, 2, 1 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 1, 3, 2, 1 };

    std::vector<float> input0Values
        {
            0.0f,
            1.0f,

            2.0f,
            3.0f,

            4.0f,
            5.0f,
        };
    std::vector<float> input1Values
        {
            0.5f
        };
    // Set output data
    std::vector<float> expectedOutputValues
        {
            0.5f,
            1.5f,

            2.5f,
            3.5f,

            4.5f,
            5.5f,
        };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_ADD,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues,
                                 1.0f,
                                 0,
                                 true);
}

void AddActivationTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 1 };
    std::vector<int32_t> input1Shape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<float> input0Values { 4.0f, 0.8f, 0.7f, -0.8f };
    std::vector<float> input1Values { 0.7f, -1.2f, 0.8f, 0.5f };
    std::vector<float> expectedOutputValues { 4.7f, 0.0f, 1.5f, 0.0f };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_ADD,
                                 tflite::ActivationFunctionType_RELU,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void AddUint8Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 3 };
    std::vector<int32_t> input1Shape { 1, 2, 2, 3 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 3 };

    std::vector<uint8_t> input0Values =
    {
        63,  35,  77,  70,  56, 112,
        203,  28, 252, 168, 245,  91
    };

    std::vector<uint8_t> input1Values =
    {
        21,   7, 175, 231, 175, 210,
        126, 161,  63,  21, 105, 126
    };

    std::vector<uint8_t> expectedOutputValues =
    {
        81,  39, 249, 255, 228, 255,
        255, 186, 255, 186, 255, 214,
    };

    ElementwiseBinaryTest<uint8_t>(tflite::BuiltinOperator_ADD,
                                   tflite::ActivationFunctionType_NONE,
                                   ::tflite::TensorType_UINT8,
                                   backends,
                                   input0Shape,
                                   input1Shape,
                                   expectedOutputShape,
                                   input0Values,
                                   input1Values,
                                   expectedOutputValues, 7.0f, 3);
}

void DivFP32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 2, 2, 2, 2 };
    std::vector<int32_t> input1Shape { 2, 2, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 2, 2, 2, 2 };

    std::vector<float> input0Values =
    {
        2.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 3.f,
        4.f, 4.f, 4.f, 4.f, 5.f, 5.f, 5.f, 5.f

    };

    std::vector<float> input1Values =
    {
        1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f,
        4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f
    };

    std::vector<float> expectedOutputValues =
    {
        2.f, 2.f, 2.f, 2.f, 1.50f, 1.50f, 1.50f, 1.50f,
        1.f, 1.f, 1.f, 1.f, 1.25f, 1.25f, 1.25f, 1.25f
    };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_DIV,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void DivBroadcastTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 1, 1, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 2 };

    std::vector<float> input0Values = { 2, 4, 6, 8, 10, 12, 14, 16 };
    std::vector<float> input1Values = { 2 };
    std::vector<float> expectedOutputValues = { 1, 2, 3, 4, 5, 6, 7, 8 };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_DIV,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void DivUint8Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 2, 2, 2, 2 };
    std::vector<int32_t> input1Shape { 2, 2, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 2, 2, 2, 2 };

    std::vector<uint8_t> input0Values =
    {
        2, 2, 2, 2,  3, 3, 3, 3,
        4, 4, 4, 4,  5, 5, 5, 5

    };

    std::vector<uint8_t> input1Values =
    {
        1, 1, 1, 1,  2, 2, 2, 2,
        4, 4, 4, 4,  4, 4, 4, 4
    };

    std::vector<uint8_t> expectedOutputValues =
    {
        8, 8, 8, 8,  6, 6, 6, 6,
        4, 4, 4, 4,  5, 5, 5, 5
    };

    ElementwiseBinaryTest<uint8_t>(tflite::BuiltinOperator_DIV,
                                   tflite::ActivationFunctionType_NONE,
                                   ::tflite::TensorType_UINT8,
                                   backends,
                                   input0Shape,
                                   input1Shape,
                                   expectedOutputShape,
                                   input0Values,
                                   input1Values,
                                   expectedOutputValues, 0.25f, 0);
}

void FloorDivFP32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 2, 2, 2, 2 };
    std::vector<int32_t> input1Shape { 2, 2, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 2, 2, 2, 2 };

    std::vector<float> input0Values =
    {
        -37.5f, -15.2f, -8.76f, -2.0f,  -2.6f, -1.0f,  -0.8f,   0.0f,
          4.0f,   1.6f,  2.0f,   5.2f,   6.0f, 35.04f, 60.8f, 150.0f
    };

    std::vector<float> input1Values =
    {
        1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f,
        4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f
    };

    std::vector<float> expectedOutputValues =
    {
        -38.0f, -16.0f, -9.0f,  -2.0f, -2.0f, -1.0f,  -1.0f,  0.0f,
          1.0f,   0.0f,  0.0f,   1.0f,  1.0f,  8.0f,  15.0f, 37.0f
    };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_FLOOR_DIV,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);

}

void MaxFP32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 2, 2, 2, 2 };
    std::vector<int32_t> input1Shape { 2, 2, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 2, 2, 2, 2 };

    std::vector<float> input0Values =
    {
        1.f, 1.f, 5.f, 1.f,  2.f, 2.f, 7.f, 2.f,
        3.f, 3.f, 3.f, 3.f,  4.f, 4.f, 4.f, 4.f

    };

    std::vector<float> input1Values =
    {
        2.f, 2.f, 2.f, 2.f,  3.f, 3.f, 3.f, 3.f,
        4.f, 4.f, 4.f, 4.f,  5.f, 5.f, 5.f, 5.f
    };

    std::vector<float> expectedOutputValues =
    {
        2.f,  2.f, 5.f,  2.f,   3.f,  3.f,  7.f,  3.f,
        4.f, 4.f, 4.f, 4.f,  5.f, 5.f, 5.f, 5.f
    };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_MAXIMUM,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void MaxBroadcastTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 1, 1, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 2 };

    std::vector<float> input0Values = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };
    std::vector<float> input1Values = { 4.f };
    std::vector<float> expectedOutputValues = { 4.f, 4.f, 4.f, 4.f, 5.f, 6.f, 7.f, 8.f };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_MAXIMUM,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void MaxUint8Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 2, 2, 2, 2 };
    std::vector<int32_t> input1Shape { 2, 2, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 2, 2, 2, 2 };

    std::vector<uint8_t> input0Values =
    {
        1, 1, 1, 1, 7, 8, 9, 9,
        3, 3, 3, 3, 4, 4, 4, 4

    };

    std::vector<uint8_t> input1Values =
    {
        2, 2, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 5, 5, 5, 5
    };

    std::vector<uint8_t> expectedOutputValues =
    {
        2, 2, 2, 2, 7, 8, 9, 9,
        4, 4, 4, 4, 5, 5, 5, 5
    };

    ElementwiseBinaryTest<uint8_t>(tflite::BuiltinOperator_MAXIMUM,
                                   tflite::ActivationFunctionType_NONE,
                                   ::tflite::TensorType_UINT8,
                                   backends,
                                   input0Shape,
                                   input1Shape,
                                   expectedOutputShape,
                                   input0Values,
                                   input1Values,
                                   expectedOutputValues, 1.0f, 0);
}

void MinFP32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 2, 2, 2, 2 };
    std::vector<int32_t> input1Shape { 2, 2, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 2, 2, 2, 2 };

    std::vector<float> input0Values =
    {
        1.f, 1.f, 5.f, 1.f,  2.f, 2.f, 7.f, 2.f,
        3.f, 3.f, 3.f, 3.f,  4.f, 4.f, 4.f, 4.f

    };

    std::vector<float> input1Values =
    {
        2.f, 2.f, 2.f, 2.f,  3.f, 3.f, 3.f, 3.f,
        1.f, 1.f, 1.f, 1.f,  5.f, 5.f, 5.f, 5.f
    };

    std::vector<float> expectedOutputValues =
    {
        1.f,  1.f, 2.f,  1.f,   2.f,  2.f,  3.f,  2.f,
        1.f, 1.f, 1.f, 1.f,  4.f, 4.f, 4.f, 4.f
    };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_MINIMUM,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void MinBroadcastTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 1, 1, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 2 };

    std::vector<float> input0Values = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };

    std::vector<float> input1Values = { 4.f };

    std::vector<float> expectedOutputValues = { 1.f, 2.f, 3.f, 4.f, 4.f, 4.f, 4.f, 4.f };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_MINIMUM,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void MinUint8Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 2, 2, 2, 2 };
    std::vector<int32_t> input1Shape { 2, 2, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 2, 2, 2, 2 };

    std::vector<uint8_t> input0Values =
    {
        1, 1, 1, 1, 7, 8, 9, 9,
        3, 3, 3, 3, 4, 4, 4, 4

    };

    std::vector<uint8_t> input1Values =
    {
        2, 2, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 5, 5, 5, 5
    };

    std::vector<uint8_t> expectedOutputValues =
    {
        1, 1, 1, 1, 3, 3, 3, 3,
        3, 3, 3, 3, 4, 4, 4, 4
    };

    ElementwiseBinaryTest<uint8_t>(tflite::BuiltinOperator_MINIMUM,
                                   tflite::ActivationFunctionType_NONE,
                                   ::tflite::TensorType_UINT8,
                                   backends,
                                   input0Shape,
                                   input1Shape,
                                   expectedOutputShape,
                                   input0Values,
                                   input1Values,
                                   expectedOutputValues, 1.0f, 0);
}

void MulFP32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 2, 2, 2, 2 };
    std::vector<int32_t> input1Shape { 2, 2, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 2, 2, 2, 2 };

    std::vector<float> input0Values =
    {
        1.f, 1.f, 1.f, 1.f,  2.f, 2.f, 2.f, 2.f,
        3.f, 3.f, 3.f, 3.f,  4.f, 4.f, 4.f, 4.f

    };

    std::vector<float> input1Values =
    {
        2.f, 2.f, 2.f, 2.f,  3.f, 3.f, 3.f, 3.f,
        4.f, 4.f, 4.f, 4.f,  5.f, 5.f, 5.f, 5.f
    };

    std::vector<float> expectedOutputValues =
    {
        2.f,  2.f,  2.f,  2.f,   6.f,  6.f,  6.f,  6.f,
        12.f, 12.f, 12.f, 12.f,  20.f, 20.f, 20.f, 20.f
    };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_MUL,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void MulBroadcastTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 1, 1, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 2 };

    std::vector<float> input0Values = { 2, 4, 6, 8, 10, 12, 14, 16 };
    std::vector<float> input1Values = { 2 };
    std::vector<float> expectedOutputValues = { 4, 8, 12, 16, 20, 24, 28, 32 };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_MUL,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void MulUint8Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 3 };
    std::vector<int32_t> input1Shape { 1, 1, 1, 3 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 3 };

    std::vector<uint8_t> input0Values =
    {
        1, 2, 3,    4,  5,  6,
        7, 8, 9,   10, 11, 12

    };

    std::vector<uint8_t> input1Values = { 1, 2, 3 };

    std::vector<uint8_t> expectedOutputValues =
    {
        1,  4,   9,     4, 10, 18,
        7, 16,  27,    10, 22, 36
    };

    ElementwiseBinaryTest<uint8_t>(tflite::BuiltinOperator_MUL,
                                   tflite::ActivationFunctionType_NONE,
                                   ::tflite::TensorType_UINT8,
                                   backends,
                                   input0Shape,
                                   input1Shape,
                                   expectedOutputShape,
                                   input0Values,
                                   input1Values,
                                   expectedOutputValues, 1.0f, 0);
}

void MulActivationTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 1 };
    std::vector<int32_t> input1Shape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<float> input0Values { 4.0f, 0.0f, 1.0f, 0.5f };
    std::vector<float> input1Values { -2.0f, -1.2f, 2.5f, 2.0f };
    std::vector<float> expectedOutputValues { 0.0f, 0.0f, 2.5f, 1.0f };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_MUL,
                                 tflite::ActivationFunctionType_RELU,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void SubFP32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 1, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 1, 1, 2, 2 };

    std::vector<float> input0Values = { 1, 3, 3, -7 };
    std::vector<float> input1Values = { 1, -1, 0, -2 };
    std::vector<float> expectedOutputValues = { 0, 4, 3, -5 };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_SUB,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void SubBroadcastTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 1, 1, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 1, 2, 2 };

    std::vector<float> input0Values = { 2, 3, 4, 5};
    std::vector<float> input1Values = { 10 };
    std::vector<float> expectedOutputValues = { -8, -7, -6, -5 };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_SUB,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 backends,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void SubUint8Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 1, 1, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 1, 2, 2 };

    std::vector<uint8_t> input0Values = { 10, 12, 14, 16 };
    std::vector<uint8_t> input1Values = { 2 };
    std::vector<uint8_t> expectedOutputValues = { 8, 10, 12, 14 };

    ElementwiseBinaryTest<uint8_t>(tflite::BuiltinOperator_SUB,
                                   tflite::ActivationFunctionType_NONE,
                                   ::tflite::TensorType_UINT8,
                                   backends,
                                   input0Shape,
                                   input1Shape,
                                   expectedOutputShape,
                                   input0Values,
                                   input1Values,
                                   expectedOutputValues, 1.0f, 0);
}

TEST_SUITE("ElementwiseBinary_GpuAccTests")
{

TEST_CASE ("ADD_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    AddFP32Test(backends);
}

TEST_CASE ("ADD_Broadcast_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    AddBroadcastTest(backends);
}

TEST_CASE ("ADD_Activation_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    AddActivationTest(backends);
}

TEST_CASE ("ADD_UINT8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    AddUint8Test(backends);
}

TEST_CASE ("DIV_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    DivFP32Test(backends);
}

TEST_CASE ("DIV_Broadcast_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    DivBroadcastTest(backends);
}

TEST_CASE ("FLOORDIV_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    FloorDivFP32Test(backends);
}

TEST_CASE ("MAX_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    MaxFP32Test(backends);
}

TEST_CASE ("MAX_Broadcast_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    MaxBroadcastTest(backends);
}

TEST_CASE ("MAX_UINT8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    MaxUint8Test(backends);
}

TEST_CASE ("MIN_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    MinFP32Test(backends);
}

TEST_CASE ("MIN_Broadcast_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    MinBroadcastTest(backends);
}

TEST_CASE ("MIN_UINT8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    MinUint8Test(backends);
}

TEST_CASE ("MUL_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    MulFP32Test(backends);
}

TEST_CASE ("MUL_Broadcast_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    MulBroadcastTest(backends);
}

TEST_CASE ("MUL_Activation_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    MulActivationTest(backends);
}

TEST_CASE ("MUL_UINT8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    MulUint8Test(backends);
}

TEST_CASE ("SUB_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    SubFP32Test(backends);
}

TEST_CASE ("SUB_Broadcast_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    SubBroadcastTest(backends);
}

TEST_CASE ("SUB_UINT8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    SubUint8Test(backends);
}

} //TEST_SUITE("ElementwiseBinary_GpuAccTests")



TEST_SUITE("ElementwiseBinary_CpuAccTests")
{

TEST_CASE ("ADD_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    AddFP32Test(backends);
}

TEST_CASE ("ADD_Broadcast_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    AddBroadcastTest(backends);
}

TEST_CASE ("ADD_Activation_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    AddActivationTest(backends);
}

TEST_CASE ("ADD_UINT8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    AddUint8Test(backends);
}

TEST_CASE ("DIV_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    DivFP32Test(backends);
}

TEST_CASE ("DIV_Broadcast_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    DivBroadcastTest(backends);
}

TEST_CASE ("FLOORDIV_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    FloorDivFP32Test(backends);
}

TEST_CASE ("MAX_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    MaxFP32Test(backends);
}

TEST_CASE ("MAX_Broadcast_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    MaxBroadcastTest(backends);
}

TEST_CASE ("MAX_UINT8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    MaxUint8Test(backends);
}

TEST_CASE ("MIN_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    MinFP32Test(backends);
}

TEST_CASE ("MIN_Broadcast_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    MinBroadcastTest(backends);
}

TEST_CASE ("MIN_UINT8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    MinUint8Test(backends);
}

TEST_CASE ("MUL_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    MulFP32Test(backends);
}

TEST_CASE ("MUL_Broadcast_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    MulBroadcastTest(backends);
}

TEST_CASE ("MUL_Actiation_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    MulActivationTest(backends);
}

TEST_CASE ("MUL_UINT8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    MulUint8Test(backends);
}

TEST_CASE ("SUB_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    SubFP32Test(backends);
}

TEST_CASE ("SUB_Broadcast_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    SubBroadcastTest(backends);
}

TEST_CASE ("SUB_UINT8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    SubUint8Test(backends);
}

} // TEST_SUITE("ElementwiseBinary_CpuAccTests")


TEST_SUITE("ElementwiseBinary_CpuRefTests")
{

TEST_CASE ("ADD_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    AddFP32Test(backends);
}

TEST_CASE ("ADD_Broadcast_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    AddBroadcastTest(backends);
}

TEST_CASE ("ADD_Constant_Input_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    AddConstInputTest(backends);
}

TEST_CASE ("ADD_Activation_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    AddActivationTest(backends);
}

TEST_CASE ("ADD_UINT8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    AddUint8Test(backends);
}

TEST_CASE ("DIV_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    DivFP32Test(backends);
}

TEST_CASE ("DIV_Broadcast_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    DivBroadcastTest(backends);
}

TEST_CASE ("FLOORDIV_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    FloorDivFP32Test(backends);
}

TEST_CASE ("DIV_UINT8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    DivUint8Test(backends);
}

TEST_CASE ("MAX_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MaxFP32Test(backends);
}

TEST_CASE ("MAX_Broadcast_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MaxBroadcastTest(backends);
}

TEST_CASE ("MAX_UINT8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MaxUint8Test(backends);
}

TEST_CASE ("MIN_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MinFP32Test(backends);
}

TEST_CASE ("MIN_Broadcast_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MinBroadcastTest(backends);
}

TEST_CASE ("MIN_UINT8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MinUint8Test(backends);
}

TEST_CASE ("MUL_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MulFP32Test(backends);
}

TEST_CASE ("MUL_Broadcast_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MulBroadcastTest(backends);
}

TEST_CASE ("MUL_Actiation_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MulActivationTest(backends);
}

TEST_CASE ("MUL_UINT8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MulUint8Test(backends);
}

TEST_CASE ("SUB_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    SubFP32Test(backends);
}

TEST_CASE ("SUB_Broadcast_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    SubBroadcastTest(backends);
}

TEST_CASE ("SUB_UINT8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    SubUint8Test(backends);
}

} // TEST_SUITE("ElementwiseBinary_CpuRefTests")

} // namespace armnnDelegate
