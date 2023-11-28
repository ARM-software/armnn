//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchMatMulTestHelper.hpp"

#include <flatbuffers/flatbuffers.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

TEST_SUITE("BATCH_MATMUL_Tests")
{
    TEST_CASE("BatchMatMul2DFp32SimpleTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 2, 2 };
        std::vector<int32_t> RHSInputShape{ 2, 2 };
        std::vector<int32_t> outputShape{ 2, 2 };

        std::vector<float> LHSInputValues = { 1, 2, 3, 4 };

        std::vector<float> RHSInputValues = { 5, 6, 7, 8 };

        std::vector<float> expectedOutputValues = { 19, 22, 43, 50 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_FLOAT32, LHSInputShape,
                               RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                               false);
    }

    TEST_CASE("BatchMatMul2DInt8SimpleTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 2, 2 };
        std::vector<int32_t> RHSInputShape{ 2, 2 };
        std::vector<int32_t> outputShape{ 2, 2 };

        std::vector<int8_t> LHSInputValues = { 1, 2, 3, 4 };

        std::vector<int8_t> RHSInputValues = { 5, 6, 7, 8 };

        std::vector<int8_t> expectedOutputValues = { 19, 22, 43, 50 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_INT8, LHSInputShape,
                                RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                                false);
    }

    TEST_CASE("BatchMatMul3DFp32SimpleTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 1, 2, 2 };
        std::vector<int32_t> RHSInputShape{ 1, 2, 2 };
        std::vector<int32_t> outputShape{ 1, 2, 2 };

        std::vector<float> LHSInputValues = { 1, 2, 3, 4 };

        std::vector<float> RHSInputValues = { 5, 6, 7, 8 };

        std::vector<float> expectedOutputValues = { 19, 22, 43, 50 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_FLOAT32, LHSInputShape,
                               RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                               false);
    }

    TEST_CASE("BatchMatMul3DInt8SimpleTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 1, 2, 2 };
        std::vector<int32_t> RHSInputShape{ 1, 2, 2 };
        std::vector<int32_t> outputShape{ 1, 2, 2 };

        std::vector<int8_t> LHSInputValues = { 1, 2, 3, 4 };

        std::vector<int8_t> RHSInputValues = { 5, 6, 7, 8 };

        std::vector<int8_t> expectedOutputValues = { 19, 22, 43, 50 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_INT8, LHSInputShape,
                                RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                                false);
    }

    TEST_CASE("BatchMatMul4DFp32SimpleTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 1, 1, 2, 2 };
        std::vector<int32_t> RHSInputShape{ 1, 1, 2, 2 };
        std::vector<int32_t> outputShape{ 1, 1, 2, 2 };

        std::vector<float> LHSInputValues = { 1, 2, 3, 4 };

        std::vector<float> RHSInputValues = { 5, 6, 7, 8 };

        std::vector<float> expectedOutputValues = { 19, 22, 43, 50 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_FLOAT32, LHSInputShape,
                               RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                               false);
    }

    TEST_CASE("BatchMatMul4DInt8SimpleTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 1, 1, 2, 2 };
        std::vector<int32_t> RHSInputShape{ 1, 1, 2, 2 };
        std::vector<int32_t> outputShape{ 1, 1, 2, 2 };

        std::vector<int8_t> LHSInputValues = { 1, 2, 3, 4 };

        std::vector<int8_t> RHSInputValues = { 5, 6, 7, 8 };

        std::vector<int8_t> expectedOutputValues = { 19, 22, 43, 50 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_INT8, LHSInputShape,
                                RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                                false);
    }

    TEST_CASE("BatchMatMul3DFp32BatchTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 2, 2, 2 };
        std::vector<int32_t> RHSInputShape{ 2, 2, 2 };
        std::vector<int32_t> outputShape{ 2, 2, 2 };

        std::vector<float> LHSInputValues = { 1, 2,  3,  4,

                                              9, 10, 11, 12 };

        std::vector<float> RHSInputValues = { 5,  6,  7,  8,

                                              13, 14, 15, 16 };

        std::vector<float> expectedOutputValues = { 19,  22,  43,  50,

                                                    267, 286, 323, 346 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_FLOAT32, LHSInputShape,
                               RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                               false);
    }

    TEST_CASE("BatchMatMul3DInt8BatchTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 2, 2, 2 };
        std::vector<int32_t> RHSInputShape{ 2, 2, 2 };
        std::vector<int32_t> outputShape{ 2, 2, 2 };

        std::vector<int8_t> LHSInputValues = { 1, 2,  3,  4,

                                               9, 10, 11, 12 };

        std::vector<int8_t> RHSInputValues = { 5, 6, 7, 8,

                                               1, 2, 3, 4 };

        std::vector<int8_t> expectedOutputValues = { 19, 22, 43, 50,

                                                     39, 58, 47, 70 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_INT8, LHSInputShape,
                                RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                                false);
    }

    TEST_CASE("BatchMatMul3DFp32BroadcastTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 2, 2, 2 };
        std::vector<int32_t> RHSInputShape{ 2, 2 };
        std::vector<int32_t> outputShape{ 2, 2, 2 };

        std::vector<float> LHSInputValues = { 1, 2,  3,  4,

                                              9, 10, 11, 12 };

        std::vector<float> RHSInputValues = { 13, 14, 15, 16 };

        std::vector<float> expectedOutputValues = { 43,  46,  99,  106,

                                                    267, 286, 323, 346 };

        // We know that this is only supported on CpuRef. To enable on all backends just remoev the last parameter.
        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_FLOAT32, LHSInputShape,
                               RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                               false, 1.0f, 0,{armnn::Compute::CpuRef});
    }

    TEST_CASE("BatchMatMul3DInt8BroadcastTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 2, 2, 2 };
        std::vector<int32_t> RHSInputShape{ 2, 2 };
        std::vector<int32_t> outputShape{ 2, 2, 2 };

        std::vector<int8_t> LHSInputValues = { 1, 2,  3,  4,

                                               9, 10, 11, 12 };

        std::vector<int8_t> RHSInputValues = { 1, 2, 3, 4 };

        std::vector<int8_t> expectedOutputValues = { 7,  10, 15, 22,

                                                     39, 58, 47, 70 };

        // We know that this is only supported on CpuRef. To enable on all backends just remoev the last parameter.
        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_INT8, LHSInputShape,
                                RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                                false, 1.0f, 0,{armnn::Compute::CpuRef});
    }

    TEST_CASE("BatchMatMul3D2DFp32BroadcastTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 2, 2, 2 };
        std::vector<int32_t> RHSInputShape{ 2, 2 };
        std::vector<int32_t> outputShape{ 2, 2, 2 };

        std::vector<float> LHSInputValues = { 1, 2,  3,  4,

                                              9, 10, 11, 12 };

        std::vector<float> RHSInputValues = { 13, 14, 15, 16 };

        std::vector<float> expectedOutputValues = { 43,  46,  99,  106,

                                                    267, 286, 323, 346 };

        // We know that this is only supported on CpuRef. To enable on all backends just remoev the last parameter.
        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_FLOAT32, LHSInputShape,
                               RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                               false, 1.0f, 0,{armnn::Compute::CpuRef});
    }

    TEST_CASE("BatchMatMul3D2DInt8BroadcastTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 2, 2, 2 };
        std::vector<int32_t> RHSInputShape{ 2, 2 };
        std::vector<int32_t> outputShape{ 2, 2, 2 };

        std::vector<int8_t> LHSInputValues = { 1, 2,  3,  4,

                                               9, 10, 11, 12 };

        std::vector<int8_t> RHSInputValues = { 1, 2, 3, 4 };

        std::vector<int8_t> expectedOutputValues = { 7,  10, 15, 22,

                                                     39, 58, 47, 70 };

        // We know that this is only supported on CpuRef. To enable on all backends just remoev the last parameter.
        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_INT8, LHSInputShape,
                                RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                                false, 1.0f, 0,{armnn::Compute::CpuRef});
    }

    TEST_CASE("BatchMatMul2DFp32TinyTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 1, 1 };
        std::vector<int32_t> RHSInputShape{ 1, 1 };
        std::vector<int32_t> outputShape{ 1, 1 };

        std::vector<float> LHSInputValues = { 3 };

        std::vector<float> RHSInputValues = { 5 };

        std::vector<float> expectedOutputValues = { 15 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_FLOAT32, LHSInputShape,
                               RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                               false);
    }

    TEST_CASE("BatchMatMul2DInt8TinyTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 1, 1 };
        std::vector<int32_t> RHSInputShape{ 1, 1 };
        std::vector<int32_t> outputShape{ 1, 1 };

        std::vector<int8_t> LHSInputValues = { 3 };

        std::vector<int8_t> RHSInputValues = { 5 };

        std::vector<int8_t> expectedOutputValues = { 15 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_INT8, LHSInputShape,
                                RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                                false);
    }

    TEST_CASE("BatchMatMulNonSquareFp32Test")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 2, 5, 3 };
        std::vector<int32_t> RHSInputShape{ 2, 3, 4 };
        std::vector<int32_t> outputShape{ 2, 5, 4 };

        std::vector<float> LHSInputValues = { 8, 8, 4, 6, 1, 3, 8, 8, 3, 8, 9, 8, 5, 4, 4,

                                              1, 8, 5, 7, 1, 1, 8, 7, 9, 3, 2, 7, 8, 5, 3 };

        std::vector<float> RHSInputValues = { 6, 2, 3, 2, 6, 2, 2, 8, 3, 7, 8, 1,

                                              7, 2, 9, 5, 2, 3, 1, 3, 2, 7, 7, 5 };

        std::vector<float> expectedOutputValues = { 108, 60,  72,  84, 51,  35, 44, 23, 105, 53,
                                                    64,  83,  126, 90, 106, 96, 66, 46, 55,  46,

                                                    33,  61,  52,  54, 53,  24, 71, 43, 88,  100,
                                                    142, 106, 39,  61, 78,  56, 72, 52, 98,  70 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_FLOAT32, LHSInputShape,
                               RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                               false);
    }

    TEST_CASE("BatchMatMulNonSquareInt8Test")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 2, 5, 3 };
        std::vector<int32_t> RHSInputShape{ 2, 3, 4 };
        std::vector<int32_t> outputShape{ 2, 5, 4 };

        std::vector<int8_t> LHSInputValues = { 8, 8, 4, 6, 1, 3, 8, 8, 3, 8, 9, 8, 5, 4, 4,

                                               1, 8, 5, 7, 1, 1, 8, 7, 9, 3, 2, 7, 8, 5, 3 };

        std::vector<int8_t> RHSInputValues = { 6, 2, 3, 2, 6, 2, 2, 8, 3, 7, 8, 1,

                                               7, 2, 3, 5, 2, 3, 1, 3, 2, 7, 7, 5 };

        std::vector<int8_t> expectedOutputValues = { 108, 60,  72,  84, 51,  35, 44, 23, 105, 53,
                                                     64,  83,  126, 90, 106, 96, 66, 46, 55,  46,

                                                     33,  61,  46,  54, 53,  24, 29, 43, 88,  100,
                                                     94,  106, 39,  61, 60,  56, 72, 52, 50,  70 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_INT8, LHSInputShape,
                                RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, false,
                                false);
    }

    TEST_CASE("BatchMatMul2DFp32SimpleAdjointTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 3, 3 };
        std::vector<int32_t> RHSInputShape{ 3, 3 };
        std::vector<int32_t> outputShape{ 3, 3 };

        std::vector<float> LHSInputValues = { 3, 1, 1, 1, 3, -1, 2, 4, 1 };

        std::vector<float> RHSInputValues = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

        std::vector<float> expectedOutputValues = { 3, 1, 2, 1, 3, 4, 1, -1, 1 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_FLOAT32, LHSInputShape,
                               RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, true,
                               false);
    }

    TEST_CASE("BatchMatMul2DInt8SimpleAdjointTest")
    {
        // Set input data
        std::vector<int32_t> LHSInputShape{ 3, 3 };
        std::vector<int32_t> RHSInputShape{ 3, 3 };
        std::vector<int32_t> outputShape{ 3, 3 };

        std::vector<int8_t> LHSInputValues = { 3, 1, 1, 1, 3, -1, 2, 4, 1 };

        std::vector<int8_t> RHSInputValues = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

        std::vector<int8_t> expectedOutputValues = { 3, 1, 2, 1, 3, 4, 1, -1, 1 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL, ::tflite::TensorType_INT8, LHSInputShape,
                                RHSInputShape, outputShape, LHSInputValues, RHSInputValues, expectedOutputValues, true,
                                false);
    }
}
}