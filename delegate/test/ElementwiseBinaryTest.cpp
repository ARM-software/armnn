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

#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void AddFP32Test()
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
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void AddBroadcastTest()
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
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void AddConstInputTest()
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

void AddActivationTest()
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
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void AddUint8Test()
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
                                   input0Shape,
                                   input1Shape,
                                   expectedOutputShape,
                                   input0Values,
                                   input1Values,
                                   expectedOutputValues, 7.0f, 3);
}

void DivFP32Test()
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
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void DivBroadcastTest()
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
                                   input0Shape,
                                   input1Shape,
                                   expectedOutputShape,
                                   input0Values,
                                   input1Values,
                                   expectedOutputValues,
                                   0.25f,
                                   0,
                                   false,
                                   backends);
}

void FloorDivFP32Test()
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
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);

}

void MaxFP32Test()
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
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void MaxBroadcastTest()
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
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void MaxUint8Test()
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
                                   input0Shape,
                                   input1Shape,
                                   expectedOutputShape,
                                   input0Values,
                                   input1Values,
                                   expectedOutputValues, 1.0f, 0);
}

void MinFP32Test()
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
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void MinBroadcastTest()
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
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void MinUint8Test()
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
                                   input0Shape,
                                   input1Shape,
                                   expectedOutputShape,
                                   input0Values,
                                   input1Values,
                                   expectedOutputValues, 1.0f, 0);
}

void MulFP32Test()
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
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void MulBroadcastTest()
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
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void MulUint8Test()
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
                                   input0Shape,
                                   input1Shape,
                                   expectedOutputShape,
                                   input0Values,
                                   input1Values,
                                   expectedOutputValues, 1.0f, 0);
}

void MulActivationTest()
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
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void SubFP32Test()
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
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void PowerFP32Test()
{
    std::vector<int32_t> input0Shape { 1, 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 1, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 1, 1, 2, 2 };

    std::vector<float> input0Values = { 1, 3, 3, -7 };
    std::vector<float> input1Values = { 1, 1, 0, 2 };
    std::vector<float> expectedOutputValues = { 1, 3, 1, 49 };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_POW,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void SqDiffFP32Test()
{
    std::vector<int32_t> input0Shape { 1, 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1, 1, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 1, 1, 2, 2 };

    std::vector<float> input0Values = { 1, 3, 3, -7 };
    std::vector<float> input1Values = { 1, -1, 0, -2 };
    std::vector<float> expectedOutputValues = { 0, 16, 9, 25 };

    ElementwiseBinaryTest<float>(tflite::BuiltinOperator_SQUARED_DIFFERENCE,
                                 tflite::ActivationFunctionType_NONE,
                                 ::tflite::TensorType_FLOAT32,
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void SubBroadcastTest()
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
                                 input0Shape,
                                 input1Shape,
                                 expectedOutputShape,
                                 input0Values,
                                 input1Values,
                                 expectedOutputValues);
}

void SubUint8Test()
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
                                   input0Shape,
                                   input1Shape,
                                   expectedOutputShape,
                                   input0Values,
                                   input1Values,
                                   expectedOutputValues, 1.0f, 0);
}

TEST_SUITE("ElementwiseBinary_Tests")
{

TEST_CASE ("ADD_FP32_Test")
{
    AddFP32Test();
}

TEST_CASE ("ADD_Broadcast_Test")
{
    AddBroadcastTest();
}

TEST_CASE ("ADD_Constant_Input_Test")
{
    AddConstInputTest();
}

TEST_CASE ("ADD_Activation_Test")
{
    AddActivationTest();
}

TEST_CASE ("ADD_UINT8_Test")
{
    AddUint8Test();
}

TEST_CASE ("DIV_FP32_Test")
{
    DivFP32Test();
}

TEST_CASE ("DIV_Broadcast_Test")
{
    DivBroadcastTest();
}

TEST_CASE ("FLOORDIV_FP32_Test")
{
    FloorDivFP32Test();
}

TEST_CASE ("DIV_UINT8_Test")
{
    // Only works on CpuRef.
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    DivUint8Test(backends);
}

TEST_CASE ("MAX_FP32_Test")
{
    MaxFP32Test();
}

TEST_CASE ("MAX_Broadcast_Test")
{
    MaxBroadcastTest();
}

TEST_CASE ("MAX_UINT8_Test")
{
    MaxUint8Test();
}

TEST_CASE ("MIN_FP32_Test")
{
    MinFP32Test();
}

TEST_CASE ("MIN_Broadcast_Test")
{
    MinBroadcastTest();
}

TEST_CASE ("MIN_UINT8_Test")
{
    MinUint8Test();
}

TEST_CASE ("MUL_FP32_Test")
{
    MulFP32Test();
}

TEST_CASE ("MUL_Broadcast_Test")
{
    MulBroadcastTest();
}

TEST_CASE ("MUL_Actiation_Test")
{
    MulActivationTest();
}

TEST_CASE ("MUL_UINT8_Test")
{
    MulUint8Test();
}

TEST_CASE ("SUB_FP32_Test")
{
    SubFP32Test();
}

TEST_CASE ("SUB_Broadcast_Test")
{
    SubBroadcastTest();
}

TEST_CASE ("SUB_UINT8_Test")
{
    SubUint8Test();
}

TEST_CASE ("SqDiffFP32_Test")
{
    SqDiffFP32Test();
}

TEST_CASE ("PowerFP32_Test")
{
    PowerFP32Test();
}

} // TEST_SUITE("ElementwiseBinary_CpuRefTests")

} // namespace armnnDelegate
