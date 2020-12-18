//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RedefineTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <doctest/doctest.h>

#include <half/half.hpp>

using Half = half_float::half;

namespace armnnDelegate
{

void ReshapeSimpleTest(std::vector<armnn::BackendId>& backends, bool useOption = true)
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
                        backends,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        targetShape,
                        useOption);
}

using namespace half_float::literal;

void ReshapeSimpleFloat16Test(std::vector<armnn::BackendId>& backends, bool useOption = true)
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
                        backends,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        targetShape,
                        useOption);
}

void ReshapeReduceDimTest(std::vector<armnn::BackendId>& backends, bool useOption = true)
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
                        backends,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        targetShape,
                        useOption);
}

void ReshapeFlattenTest(std::vector<armnn::BackendId>& backends, bool useOption = true)
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
                        backends,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        targetShape,
                        useOption);
}

void ReshapeFlattenAllTest(std::vector<armnn::BackendId>& backends, bool useOption = true)
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
                        backends,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        targetShape,
                        useOption);
}

void ReshapeInt8Test(std::vector<armnn::BackendId>& backends, bool useOption = true)
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
                         backends,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         targetShape,
                         useOption,
                         2.5f,
                         1);
}

void ReshapeUint8Test(std::vector<armnn::BackendId>& backends, bool useOption = true)
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
                          backends,
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          targetShape,
                          useOption,
                          2.5f,
                          1);
}

void ReshapeInt16Test(std::vector<armnn::BackendId>& backends, bool useOption = true)
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
                          backends,
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          targetShape,
                          useOption,
                          2.5f,
                          0);
}

TEST_SUITE("Reshape_GpuAccTests")
{

TEST_CASE ("Reshape_Simple_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ReshapeSimpleTest(backends);
}

TEST_CASE ("Reshape_ReduceDimension_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ReshapeReduceDimTest(backends);
}

TEST_CASE ("Reshape_Flatten_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ReshapeFlattenTest(backends);
}

TEST_CASE ("Reshape_FlattenAll_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ReshapeFlattenAllTest(backends);
}

TEST_CASE ("Reshape_Int8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ReshapeInt8Test(backends);
}

TEST_CASE ("Reshape_Uint8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ReshapeUint8Test(backends);
}

TEST_CASE ("Reshape_Float16_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ReshapeSimpleFloat16Test(backends);
}

TEST_CASE ("Reshape_Simple_ShapeTensor_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ReshapeSimpleTest(backends, false);
}

TEST_CASE ("Reshape_ReduceDimension_ShapeTensor_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ReshapeReduceDimTest(backends, false);
}

TEST_CASE ("Reshape_Flatten_ShapeTensor_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ReshapeFlattenTest(backends, false);
}

TEST_CASE ("Reshape_FlattenAll_ShapeTensor_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ReshapeFlattenAllTest(backends, false);
}

TEST_CASE ("Reshape_Int8_ShapeTensor_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ReshapeInt8Test(backends, false);
}

TEST_CASE ("Reshape_Uint8_ShapeTensor_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ReshapeUint8Test(backends, false);
}

TEST_CASE ("Reshape_Float16_ShapeTensor_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ReshapeSimpleFloat16Test(backends, false);
}

} // TEST_SUITE("Reshape_GpuAccTests")

TEST_SUITE("Reshape_CpuAccTests")
{

TEST_CASE ("Reshape_Simple_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ReshapeSimpleTest(backends);
}

TEST_CASE ("Reshape_ReduceDimension_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ReshapeReduceDimTest(backends);
}

TEST_CASE ("Reshape_Flatten_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ReshapeFlattenTest(backends);
}

TEST_CASE ("Reshape_FlattenAll_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ReshapeFlattenAllTest(backends);
}

TEST_CASE ("Reshape_Int8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ReshapeInt8Test(backends);
}

TEST_CASE ("Reshape_Uint8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ReshapeUint8Test(backends);
}

TEST_CASE ("Reshape_Float16_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ReshapeSimpleFloat16Test(backends);
}

TEST_CASE ("Reshape_Simple_ShapeTensor_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ReshapeSimpleTest(backends, false);
}

TEST_CASE ("Reshape_ReduceDimension_ShapeTensor_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ReshapeReduceDimTest(backends, false);
}

TEST_CASE ("Reshape_Flatten_ShapeTensor_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ReshapeFlattenTest(backends, false);
}

TEST_CASE ("Reshape_FlattenAll_ShapeTensor_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ReshapeFlattenAllTest(backends, false);
}

TEST_CASE ("Reshape_Int8_ShapeTensor_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ReshapeInt8Test(backends, false);
}

TEST_CASE ("Reshape_Uint8_ShapeTensor_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ReshapeUint8Test(backends, false);
}

TEST_CASE ("Reshape_Float16_ShapeTensor_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ReshapeSimpleFloat16Test(backends, false);
}

} // TEST_SUITE("Reshape_CpuAccTests")

TEST_SUITE("Reshape_CpuRefTests")
{

TEST_CASE ("Reshape_Simple_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeSimpleTest(backends);
}

TEST_CASE ("Reshape_ReduceDimension_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeReduceDimTest(backends);
}

TEST_CASE ("Reshape_Flatten_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeFlattenTest(backends);
}

TEST_CASE ("Reshape_FlattenAll_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeFlattenAllTest(backends);
}

TEST_CASE ("Reshape_Int8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeInt8Test(backends);
}

TEST_CASE ("Reshape_Uint8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeUint8Test(backends);
}

TEST_CASE ("Reshape_Int16_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeInt16Test(backends);
}

TEST_CASE ("Reshape_Float16_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeSimpleFloat16Test(backends);
}

TEST_CASE ("Reshape_Simple_ShapeTensor_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeSimpleTest(backends, false);
}

TEST_CASE ("Reshape_ReduceDimension_ShapeTensor_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeReduceDimTest(backends, false);
}

TEST_CASE ("Reshape_Flatten_ShapeTensor_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeFlattenTest(backends, false);
}

TEST_CASE ("Reshape_FlattenAll_ShapeTensor_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeFlattenAllTest(backends, false);
}

TEST_CASE ("Reshape_Int8_ShapeTensor_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeInt8Test(backends, false);
}

TEST_CASE ("Reshape_Uint8_ShapeTensor_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeUint8Test(backends, false);
}

TEST_CASE ("Reshape_Int16_ShapeTensor_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeInt16Test(backends, false);
}

TEST_CASE ("Reshape_Float16_ShapeTensor_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ReshapeSimpleFloat16Test(backends, false);
}

} // TEST_SUITE("Reshape_CpuRefTests")

} // namespace armnnDelegate