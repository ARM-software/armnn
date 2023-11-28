//
// Copyright © 2020, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ComparisonTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void EqualFP32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 2, 2, 2, 2 };
    std::vector<int32_t> input1Shape { 2, 2, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 2, 2, 2, 2 };

    std::vector<float> input0Values =
    {
        1.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f,
        3.f, 3.f, 3.f, 3.f, 4.f, 4.f, 4.f, 4.f
    };

    std::vector<float> input1Values =
    {
        1.f, 1.f, 1.f, 1.f, 3.f, 3.f, 3.f, 3.f,
        5.f, 5.f, 5.f, 5.f, 4.f, 4.f, 4.f, 4.f
    };

    std::vector<bool> expectedOutputValues =
    {
        1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1
    };


    ComparisonTest<float>(tflite::BuiltinOperator_EQUAL,
                          ::tflite::TensorType_FLOAT32,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues,
                          1.0f,
                          0,
                          backends);

}

void EqualBroadcastTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 3 };
    std::vector<int32_t> input1Shape { 1, 1, 1, 3 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 3 };

    std::vector<float> input0Values
    {
        1.f, 2.f, 3.f,  4.f,  5.f,  6.f,
        7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    };
    std::vector<float> input1Values { 4.f, 5.f, 6.f };
    // Set output data
    std::vector<bool> expectedOutputValues
    {
        0, 0, 0, 1, 1, 1,
        0, 0, 0, 0, 0, 0
    };
    ComparisonTest<float>(tflite::BuiltinOperator_EQUAL,
                          ::tflite::TensorType_FLOAT32,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues,
                          1.0f,
                          0,
                          backends);
}

void EqualInt32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 1 };
    std::vector<int32_t> input1Shape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<int32_t> input0Values = { 1, 5, 6, 4 };

    std::vector<int32_t> input1Values = { 1, 3, 9, 4 };

    std::vector<bool> expectedOutputValues = { 1, 0, 0, 1 };

    ComparisonTest<int32_t>(tflite::BuiltinOperator_EQUAL,
                            ::tflite::TensorType_INT32,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues,
                            1.0f,
                            0,
                            backends);
}

void NotEqualFP32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 2, 2, 2, 2 };
    std::vector<int32_t> input1Shape { 2, 2, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 2, 2, 2, 2 };

    std::vector<float> input0Values =
    {
        1.f, 1.f, 1.f, 1.f, 5.f, 5.f, 5.f, 5.f,
        3.f, 3.f, 3.f, 3.f, 4.f, 4.f, 4.f, 4.f
    };

    std::vector<float> input1Values =
    {
        1.f, 1.f, 1.f, 1.f, 3.f, 3.f, 3.f, 3.f,
        5.f, 5.f, 5.f, 5.f, 4.f, 4.f, 4.f, 4.f
    };

    std::vector<bool> expectedOutputValues =
    {
        0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0
    };

    ComparisonTest<float>(tflite::BuiltinOperator_NOT_EQUAL,
                          ::tflite::TensorType_FLOAT32,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues,
                          1.0f,
                          0,
                          backends);
}

void NotEqualBroadcastTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 3 };
    std::vector<int32_t> input1Shape { 1, 1, 1, 3 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 3 };

    std::vector<float> input0Values
    {
        1.f, 2.f, 3.f,  4.f,  5.f,  6.f,
        7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    };
    std::vector<float> input1Values { 4.f, 5.f, 6.f };
    // Set output data
    std::vector<bool> expectedOutputValues
    {
        1, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 1, 1
    };
    ComparisonTest<float>(tflite::BuiltinOperator_NOT_EQUAL,
                          ::tflite::TensorType_FLOAT32,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues,
                          1.0f,
                          0,
                          backends);
}

void NotEqualInt32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 1 };
    std::vector<int32_t> input1Shape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<int32_t> input0Values = { 1, 5, 6, 4 };

    std::vector<int32_t> input1Values = { 1, 3, 9, 4 };

    std::vector<bool> expectedOutputValues = { 0, 1, 1, 0 };

    ComparisonTest<int32_t>(tflite::BuiltinOperator_NOT_EQUAL,
                            ::tflite::TensorType_INT32,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues,
                            1.0f,
                            0,
                            backends);
}

void GreaterFP32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 1 };
    std::vector<int32_t> input1Shape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<float> input0Values = { 1, 5, 6, 4 };

    std::vector<float> input1Values = { 1, 3, 9, 4 };

    std::vector<bool> expectedOutputValues = { 0, 1, 0, 0 };

    ComparisonTest<float>(tflite::BuiltinOperator_GREATER,
                          ::tflite::TensorType_FLOAT32,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues,
                          1.0f,
                          0,
                          backends);
}

void GreaterBroadcastTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 3 };
    std::vector<int32_t> input1Shape { 1, 1, 1, 3 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 3 };

    std::vector<float> input0Values
    {
        1.f, 2.f, 3.f,  4.f,  5.f,  6.f,
        7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    };
    std::vector<float> input1Values { 4.f, 5.f, 6.f };

    std::vector<bool> expectedOutputValues
    {
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1
    };
    ComparisonTest<float>(tflite::BuiltinOperator_GREATER,
                          ::tflite::TensorType_FLOAT32,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues,
                          1.0f,
                          0,
                          backends);
}

void GreaterInt32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 1 };
    std::vector<int32_t> input1Shape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<int32_t> input0Values = { 1, 5, 6, 4 };

    std::vector<int32_t> input1Values = { 1, 3, 9, 4 };

    std::vector<bool> expectedOutputValues = { 0, 1, 0, 0 };

    ComparisonTest<int32_t>(tflite::BuiltinOperator_GREATER,
                            ::tflite::TensorType_INT32,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues,
                            1.0f,
                            0,
                            backends);
}

void GreaterEqualFP32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 1 };
    std::vector<int32_t> input1Shape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<float> input0Values = { 1.f, 5.f, 6.f, 4.f };

    std::vector<float> input1Values = { 1.f, 3.f, 9.f, 4.f };

    std::vector<bool> expectedOutputValues = { true, true, false, true };

    ComparisonTest<float>(tflite::BuiltinOperator_GREATER_EQUAL,
                          ::tflite::TensorType_FLOAT32,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues,
                          1.0f,
                          0,
                          backends);
}

void GreaterEqualBroadcastTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 3 };
    std::vector<int32_t> input1Shape { 1, 1, 1, 3 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 3 };

    std::vector<float> input0Values
    {
        1.f, 2.f, 3.f,  4.f,  5.f,  6.f,
        7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    };
    std::vector<float> input1Values { 4.f, 5.f, 6.f };
    // Set output data
    std::vector<bool> expectedOutputValues
    {
        0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 1
    };

    ComparisonTest<float>(tflite::BuiltinOperator_GREATER_EQUAL,
                          ::tflite::TensorType_FLOAT32,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues,
                          1.0f,
                          0,
                          backends);
}

void GreaterEqualInt32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 1 };
    std::vector<int32_t> input1Shape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<int32_t> input0Values = { 1, 5, 6, 3 };

    std::vector<int32_t> input1Values = { 1, 3, 9, 4 };

    std::vector<bool> expectedOutputValues = { 1, 1, 0, 0 };

    ComparisonTest<int32_t>(tflite::BuiltinOperator_GREATER_EQUAL,
                            ::tflite::TensorType_INT32,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues,
                            1.0f,
                            0,
                            backends);
}

void LessFP32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 1 };
    std::vector<int32_t> input1Shape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<float> input0Values = { 1.f, 5.f, 6.f, 4.f };

    std::vector<float> input1Values = { 1.f, 3.f, 9.f, 4.f };

    std::vector<bool> expectedOutputValues = { false, false, true, false };

    ComparisonTest<float>(tflite::BuiltinOperator_LESS,
                          ::tflite::TensorType_FLOAT32,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues,
                          1.0f,
                          0,
                          backends);
}

void LessBroadcastTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 3 };
    std::vector<int32_t> input1Shape { 1, 1, 1, 3 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 3 };

    std::vector<float> input0Values
    {
        1.f, 2.f, 3.f,  4.f,  5.f,  6.f,
        7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    };
    std::vector<float> input1Values { 4.f, 5.f, 6.f };

    std::vector<bool> expectedOutputValues
    {
        true, true, true, false, false, false,
        false, false, false, false, false, false
    };

    ComparisonTest<float>(tflite::BuiltinOperator_LESS,
                          ::tflite::TensorType_FLOAT32,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues,
                          1.0f,
                          0,
                          backends);
}

void LessInt32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 1 };
    std::vector<int32_t> input1Shape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<int32_t> input0Values = { 1, 5, 6, 3 };

    std::vector<int32_t> input1Values = { 1, 3, 9, 4 };

    std::vector<bool> expectedOutputValues = { false, false, true, true };

    ComparisonTest<int32_t>(tflite::BuiltinOperator_LESS,
                            ::tflite::TensorType_INT32,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues,
                            1.0f,
                            0,
                            backends);
}

void LessEqualFP32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 1 };
    std::vector<int32_t> input1Shape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<float> input0Values = { 1.f, 5.f, 6.f, 4.f };

    std::vector<float> input1Values = { 1.f, 3.f, 9.f, 4.f };

    std::vector<bool> expectedOutputValues = { true, false, true, true };

    ComparisonTest<float>(tflite::BuiltinOperator_LESS_EQUAL,
                          ::tflite::TensorType_FLOAT32,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues,
                          1.0f,
                          0,
                          backends);
}

void LessEqualBroadcastTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 3 };
    std::vector<int32_t> input1Shape { 1, 1, 1, 3 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 3 };

    std::vector<float> input0Values
    {
        1.f, 2.f, 3.f,  4.f,  5.f,  6.f,
        7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    };
    std::vector<float> input1Values { 4.f, 5.f, 6.f };

    std::vector<bool> expectedOutputValues
    {
        true, true, true, true, true, true,
        false, false, false, false, false, false
    };

    ComparisonTest<float>(tflite::BuiltinOperator_LESS_EQUAL,
                          ::tflite::TensorType_FLOAT32,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues,
                          1.0f,
                          0,
                          backends);
}

void LessEqualInt32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 1 };
    std::vector<int32_t> input1Shape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<int32_t> input0Values = { 1, 5, 6, 3 };

    std::vector<int32_t> input1Values = { 1, 3, 9, 4 };

    std::vector<bool> expectedOutputValues = { true, false, true, true };

    ComparisonTest<int32_t>(tflite::BuiltinOperator_LESS_EQUAL,
                            ::tflite::TensorType_INT32,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues,
                            1.0f,
                            0,
                            backends);
}

TEST_SUITE("Comparison_Tests")
{

TEST_CASE ("EQUAL_FP32_Test")
{
    std::vector<armnn::BackendId> backends = { };
    EqualFP32Test(backends);
}

TEST_CASE ("EQUAL_Broadcast_Test")
{
    std::vector<armnn::BackendId> backends = { };
    EqualBroadcastTest(backends);
}

TEST_CASE ("EQUAL_INT32_Test")
{
    std::vector<armnn::BackendId> backends = { };
    EqualInt32Test(backends);
}

TEST_CASE ("NOT_EQUAL_FP32_Test")
{
    std::vector<armnn::BackendId> backends = { };
    NotEqualFP32Test(backends);
}

TEST_CASE ("NOT_EQUAL_Broadcast_Test")
{
    std::vector<armnn::BackendId> backends = { };
    NotEqualBroadcastTest(backends);
}

TEST_CASE ("NOT_EQUAL_INT32_Test")
{
    std::vector<armnn::BackendId> backends = { };
    NotEqualInt32Test(backends);
}

TEST_CASE ("GREATER_FP32_Test")
{
    std::vector<armnn::BackendId> backends = { };
    GreaterFP32Test(backends);
}

TEST_CASE ("GREATER_Broadcast_Test")
{
    std::vector<armnn::BackendId> backends = { };
    GreaterBroadcastTest(backends);
}

TEST_CASE ("GREATER_INT32_Test")
{
    std::vector<armnn::BackendId> backends = { };
    GreaterInt32Test(backends);
}

TEST_CASE ("GREATER_EQUAL_FP32_Test")
{
    std::vector<armnn::BackendId> backends = { };
    GreaterEqualFP32Test(backends);
}

TEST_CASE ("GREATER_EQUAL_Broadcast_Test")
{
    std::vector<armnn::BackendId> backends = { };
    GreaterEqualBroadcastTest(backends);
}

TEST_CASE ("GREATER_EQUAL_INT32_Test")
{
    std::vector<armnn::BackendId> backends = { };
    GreaterEqualInt32Test(backends);
}

TEST_CASE ("LESS_FP32_Test")
{
    std::vector<armnn::BackendId> backends = { };
    LessFP32Test(backends);
}

TEST_CASE ("LESS_Broadcast_Test")
{
    std::vector<armnn::BackendId> backends = { };
    LessBroadcastTest(backends);
}

TEST_CASE ("LESS_INT32_Test")
{
    std::vector<armnn::BackendId> backends = { };
    LessInt32Test(backends);
}

TEST_CASE ("LESS_EQUAL_FP32_Test")
{
    std::vector<armnn::BackendId> backends = { };
    LessEqualFP32Test(backends);
}

TEST_CASE ("LESS_EQUAL_Broadcast_Test")
{
    std::vector<armnn::BackendId> backends = { };
    LessEqualBroadcastTest(backends);
}

TEST_CASE ("LESS_EQUAL_INT32_Test")
{
    std::vector<armnn::BackendId> backends = { };
    LessEqualInt32Test(backends);
}
} // End TEST_SUITE("Comparison_Tests")

} // namespace armnnDelegate