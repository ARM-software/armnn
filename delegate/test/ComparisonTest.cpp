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
#include <schema_generated.h>
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
                          backends,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues);
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
                          backends,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues);
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
                            backends,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues);
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
                          backends,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues);
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
                          backends,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues);
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
                            backends,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues);
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
                          backends,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues);
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
                          backends,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues);
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
                            backends,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues);
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
                          backends,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues);
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
                          backends,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues);
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
                            backends,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues);
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
                          backends,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues);
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
                          backends,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues);
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
                            backends,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues);
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
                          backends,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues);
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
                          backends,
                          input0Shape,
                          input1Shape,
                          expectedOutputShape,
                          input0Values,
                          input1Values,
                          expectedOutputValues);
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
                            backends,
                            input0Shape,
                            input1Shape,
                            expectedOutputShape,
                            input0Values,
                            input1Values,
                            expectedOutputValues);
}

TEST_SUITE("Comparison_CpuRefTests")
{

TEST_CASE ("EQUAL_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    EqualFP32Test(backends);
}

TEST_CASE ("EQUAL_Broadcast_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    EqualBroadcastTest(backends);
}

TEST_CASE ("EQUAL_INT32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    EqualInt32Test(backends);
}

TEST_CASE ("NOT_EQUAL_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    NotEqualFP32Test(backends);
}

TEST_CASE ("NOT_EQUAL_Broadcast_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    NotEqualBroadcastTest(backends);
}

TEST_CASE ("NOT_EQUAL_INT32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    NotEqualInt32Test(backends);
}

TEST_CASE ("GREATER_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    GreaterFP32Test(backends);
}

TEST_CASE ("GREATER_Broadcast_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    GreaterBroadcastTest(backends);
}

TEST_CASE ("GREATER_INT32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    GreaterInt32Test(backends);
}

TEST_CASE ("GREATER_EQUAL_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    GreaterEqualFP32Test(backends);
}

TEST_CASE ("GREATER_EQUAL_Broadcast_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    GreaterEqualBroadcastTest(backends);
}

TEST_CASE ("GREATER_EQUAL_INT32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    GreaterEqualInt32Test(backends);
}

TEST_CASE ("LESS_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    LessFP32Test(backends);
}

TEST_CASE ("LESS_Broadcast_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    LessBroadcastTest(backends);
}

TEST_CASE ("LESS_INT32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    LessInt32Test(backends);
}

TEST_CASE ("LESS_EQUAL_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    LessEqualFP32Test(backends);
}

TEST_CASE ("LESS_EQUAL_Broadcast_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    LessEqualBroadcastTest(backends);
}

TEST_CASE ("LESS_EQUAL_INT32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    LessEqualInt32Test(backends);
}
} // End TEST_SUITE("Comparison_CpuRefTests")



TEST_SUITE("Comparison_GpuAccTests")
{

TEST_CASE ("EQUAL_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    EqualFP32Test(backends);
}

TEST_CASE ("EQUAL_Broadcast_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    EqualBroadcastTest(backends);
}

TEST_CASE ("EQUAL_INT32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    EqualInt32Test(backends);
}

TEST_CASE ("NOT_EQUAL_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    NotEqualFP32Test(backends);
}

TEST_CASE ("NOT_EQUAL_Broadcast_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    NotEqualBroadcastTest(backends);
}

TEST_CASE ("NOT_EQUAL_INT32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    NotEqualInt32Test(backends);
}

TEST_CASE ("GREATER_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc,
                                               armnn::Compute::CpuRef };
    GreaterFP32Test(backends);
}

TEST_CASE ("GREATER_Broadcast_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc,
                                               armnn::Compute::CpuRef };
    GreaterBroadcastTest(backends);
}

TEST_CASE ("GREATER_INT32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc,
                                               armnn::Compute::CpuRef };
    GreaterInt32Test(backends);
}

TEST_CASE ("GREATER_EQUAL_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    GreaterEqualFP32Test(backends);
}

TEST_CASE ("GREATER_EQUAL_Broadcast_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    GreaterEqualBroadcastTest(backends);
}

TEST_CASE ("GREATER_EQUAL_INT32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    GreaterEqualInt32Test(backends);
}

TEST_CASE ("LESS_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    LessFP32Test(backends);
}

TEST_CASE ("LESS_Broadcast_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    LessBroadcastTest(backends);
}

TEST_CASE ("LESS_INT32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    LessInt32Test(backends);
}

TEST_CASE ("LESS_EQUAL_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    LessEqualFP32Test(backends);
}

TEST_CASE ("LESS_EQUAL_Broadcast_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    LessEqualBroadcastTest(backends);
}

TEST_CASE ("LESS_EQUAL_INT32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    LessEqualInt32Test(backends);
}

} // End TEST_SUITE("Comparison_GpuAccTests")


TEST_SUITE("Comparison_CpuAccTests")
{

TEST_CASE ("EQUAL_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    EqualFP32Test(backends);
}

TEST_CASE ("EQUAL_Broadcast_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    EqualBroadcastTest(backends);
}

TEST_CASE ("EQUAL_INT32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    EqualInt32Test(backends);
}

TEST_CASE ("NOT_EQUAL_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    NotEqualFP32Test(backends);
}

TEST_CASE ("NOT_EQUAL_Broadcast_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    NotEqualBroadcastTest(backends);
}

TEST_CASE ("NOT_EQUAL_INT32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    NotEqualInt32Test(backends);
}

TEST_CASE ("GREATER_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    GreaterFP32Test(backends);
}

TEST_CASE ("GREATER_Broadcast_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    GreaterBroadcastTest(backends);
}

TEST_CASE ("GREATER_INT32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    GreaterInt32Test(backends);
}

TEST_CASE ("GREATER_EQUAL_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    GreaterEqualFP32Test(backends);
}

TEST_CASE ("GREATER_EQUAL_Broadcast_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    GreaterEqualBroadcastTest(backends);
}

TEST_CASE ("GREATER_EQUAL_INT32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    GreaterEqualInt32Test(backends);
}

TEST_CASE ("LESS_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    LessFP32Test(backends);
}

TEST_CASE ("LESS_Broadcast_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    LessBroadcastTest(backends);
}

TEST_CASE ("LESS_INT32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    LessInt32Test(backends);
}

TEST_CASE ("LESS_EQUAL_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    LessEqualFP32Test(backends);
}

TEST_CASE ("LESS_EQUAL_Broadcast_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    LessEqualBroadcastTest(backends);
}

TEST_CASE ("LESS_EQUAL_INT32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    LessEqualInt32Test(backends);
}

} // End TEST_SUITE("Comparison_CpuAccTests")

} // namespace armnnDelegate