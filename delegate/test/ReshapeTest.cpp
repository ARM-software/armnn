//
// Copyright © 2020, 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RedefineTestHelper.hpp"

#include <doctest/doctest.h>

#include <half/half.hpp>

using Half = half_float::half;

namespace armnnDelegate
{

void ReshapeSimpleTest(bool useOption = true, const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 3, 2, 2 };
    std::vector<int32_t> targetShape { 1, 3, 2, 2 };

    std::vector<float> inputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                       8.0f, 12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, -11.0f };

    std::vector<float> expectedOutputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                                8.0f, 12.0f, -15.0f, 2.0f,
                                                3.0f, -4.0f, -1.0f, -11.0f };

    RedefineTest<float>(tflite::BuiltinOperator_RESHAPE,
                        ::tflite::TensorType_FLOAT32,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        targetShape,
                        useOption,
                        backends);
}

using namespace half_float::literal;

void ReshapeSimpleFloat16Test(bool useOption = true, const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 3, 2, 2 };
    std::vector<int32_t> targetShape { 1, 3, 2, 2 };

    std::vector<Half> inputValues = { 5._h, -8._h, -10._h, 7._h,
                                      8._h, 12._h, -15._h, 2._h,
                                      3._h, -4._h, -1._h, -11._h };

    std::vector<Half> expectedOutputValues = { 5._h, -8._h, -10._h, 7._h,
                                               8._h, 12._h, -15._h, 2._h,
                                               3._h, -4._h, -1._h, -11._h };

    RedefineTest<Half>(tflite::BuiltinOperator_RESHAPE,
                        ::tflite::TensorType_FLOAT16,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        targetShape,
                        useOption,
                        backends);
}

void ReshapeReduceDimTest(bool useOption = true, const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 4, 3 };
    std::vector<int32_t> targetShape { 1, 4, 3 };

    std::vector<float> inputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                       8.0f, 12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, -11.0f };

    std::vector<float> expectedOutputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                                8.0f, 12.0f, -15.0f, 2.0f,
                                                3.0f, -4.0f, -1.0f, -11.0f };

    RedefineTest<float>(tflite::BuiltinOperator_RESHAPE,
                        ::tflite::TensorType_FLOAT32,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        targetShape,
                        useOption,
                        backends);
}

void ReshapeFlattenTest(bool useOption = true, const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 6, 2 };
    std::vector<int32_t> targetShape { -1, 2 };

    std::vector<float> inputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                       8.0f, 12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, -11.0f };

    std::vector<float> expectedOutputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                                8.0f, 12.0f, -15.0f, 2.0f,
                                                3.0f, -4.0f, -1.0f, -11.0f };

    RedefineTest<float>(tflite::BuiltinOperator_RESHAPE,
                        ::tflite::TensorType_FLOAT32,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        targetShape,
                        useOption,
                        backends);
}

void ReshapeFlattenAllTest(bool useOption = true, const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 12 };
    std::vector<int32_t> targetShape { -1 };

    std::vector<float> inputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                       8.0f, 12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, -11.0f };

    std::vector<float> expectedOutputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                                8.0f, 12.0f, -15.0f, 2.0f,
                                                3.0f, -4.0f, -1.0f, -11.0f };

    RedefineTest<float>(tflite::BuiltinOperator_RESHAPE,
                        ::tflite::TensorType_FLOAT32,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        targetShape,
                        useOption,
                        backends);
}

void ReshapeInt8Test(bool useOption = true, const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 6, 2 };
    std::vector<int32_t> targetShape { -1, 2 };

    std::vector<int8_t> inputValues = { -5, 8, -10, 7,
                                        8, 12, -15, 2,
                                        3, -4, -1, -11 };

    std::vector<int8_t> expectedOutputValues = { -5, 8, -10, 7,
                                                 8, 12, -15, 2,
                                                 3, -4, -1, -11 };

    RedefineTest<int8_t>(tflite::BuiltinOperator_RESHAPE,
                         ::tflite::TensorType_INT8,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         targetShape,
                         useOption,
                         backends,
                         2.5f,
                         1);
}

void ReshapeUint8Test(bool useOption = true, const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 6, 2 };
    std::vector<int32_t> targetShape { -1, 2 };

    std::vector<uint8_t> inputValues = { 5, 8, 10, 7,
                                         8, 12, 15, 2,
                                         3, 4, 1, 11 };

    std::vector<uint8_t> expectedOutputValues = { 5, 8, 10, 7,
                                                  8, 12, 15, 2,
                                                  3, 4, 1, 11 };

    RedefineTest<uint8_t>(tflite::BuiltinOperator_RESHAPE,
                          ::tflite::TensorType_UINT8,
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          targetShape,
                          useOption,
                          backends,
                          2.5f,
                          1);
}

void ReshapeInt16Test(bool useOption = true, const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 6, 2 };
    std::vector<int32_t> targetShape { -1, 2 };

    std::vector<int16_t> inputValues = { -5, 8, -10, 7,
                                         8, 12, -15, 2,
                                         3, -4, -1, -11 };

    std::vector<int16_t> expectedOutputValues = { -5, 8, -10, 7,
                                                  8, 12, -15, 2,
                                                  3, -4, -1, -11 };

    RedefineTest<int16_t>(tflite::BuiltinOperator_RESHAPE,
                          ::tflite::TensorType_INT16,
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          targetShape,
                          useOption,
                          backends,
                          2.5f,
                          0);
}

TEST_SUITE("ReshapeTests")
{

TEST_CASE ("Reshape_Simple_Test")
{
    ReshapeSimpleTest();
}

TEST_CASE ("Reshape_ReduceDimension_Test")
{
    ReshapeReduceDimTest();
}

TEST_CASE ("Reshape_Flatten_Test")
{
    ReshapeFlattenTest();
}

TEST_CASE ("Reshape_FlattenAll_Test")
{
    ReshapeFlattenAllTest();
}

TEST_CASE ("Reshape_Int8_Test")
{
    ReshapeInt8Test();
}

TEST_CASE ("Reshape_Uint8_Test")
{
    ReshapeUint8Test();
}

TEST_CASE ("Reshape_Int16_Test")
{
    ReshapeInt16Test();
}

TEST_CASE ("Reshape_Float16_Test")
{
    ReshapeSimpleFloat16Test();
}

TEST_CASE ("Reshape_Simple_ShapeTensor_Test")
{
    ReshapeSimpleTest(false);
}

TEST_CASE ("Reshape_ReduceDimension_ShapeTensor_Test")
{
    ReshapeReduceDimTest(false);
}

TEST_CASE ("Reshape_Flatten_ShapeTensor_Test")
{
    ReshapeFlattenTest(false);
}

TEST_CASE ("Reshape_FlattenAll_ShapeTensor_Test")
{
    ReshapeFlattenAllTest(false);
}

TEST_CASE ("Reshape_Int8_ShapeTensor_Test")
{
    ReshapeInt8Test(false);
}

TEST_CASE ("Reshape_Uint8_ShapeTensor_Test")
{
    ReshapeUint8Test(false);
}

TEST_CASE ("Reshape_Int16_ShapeTensor_Test")
{
    ReshapeInt16Test(false);
}

TEST_CASE ("Reshape_Float16_ShapeTensor_Test")
{
    ReshapeSimpleFloat16Test(false);
}

} // TEST_SUITE("ReshapeTests")

} // namespace armnnDelegate