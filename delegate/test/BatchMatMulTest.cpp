//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchMatMulTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

    void BatchMatMul2DFp32SimpleTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 2, 2 };
        std::vector<int32_t> RHSInputShape { 2, 2 };
        std::vector<int32_t> outputShape   { 2, 2 };

        std::vector<float> LHSInputValues = { 1, 2,
                                              3, 4 };

        std::vector<float> RHSInputValues = { 5, 6,
                                              7, 8  };

        std::vector<float> expectedOutputValues = { 19, 22,
                                                    43, 50 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_FLOAT32,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               false,
                               false);
    }
    void BatchMatMul2DInt8SimpleTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 2, 2 };
        std::vector<int32_t> RHSInputShape { 2, 2 };
        std::vector<int32_t> outputShape   { 2, 2 };

        std::vector<int8_t> LHSInputValues = { 1, 2,
                                              3, 4 };

        std::vector<int8_t> RHSInputValues = { 5, 6,
                                              7, 8  };

        std::vector<int8_t> expectedOutputValues = { 19, 22,
                                                    43, 50 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_INT8,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               false,
                               false);
    }

    void BatchMatMul3DFp32SimpleTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 1,2,2 };
        std::vector<int32_t> RHSInputShape { 1,2,2 };
        std::vector<int32_t> outputShape   { 1,2,2 };

        std::vector<float> LHSInputValues = { 1, 2,
                                              3, 4 };

        std::vector<float> RHSInputValues = { 5, 6,
                                              7, 8  };

        std::vector<float> expectedOutputValues = { 19, 22,
                                                    43, 50 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_FLOAT32,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               false,
                               false);
    }

    void BatchMatMul3DInt8SimpleTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 1,2,2 };
        std::vector<int32_t> RHSInputShape { 1,2,2 };
        std::vector<int32_t> outputShape   { 1,2,2 };

        std::vector<int8_t> LHSInputValues = { 1, 2,
                                              3, 4 };

        std::vector<int8_t> RHSInputValues = { 5, 6,
                                              7, 8  };

        std::vector<int8_t> expectedOutputValues = { 19, 22,
                                                    43, 50 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_INT8,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               false,
                               false);
    }

    void BatchMatMul4DFp32SimpleTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 1,1,2,2 };
        std::vector<int32_t> RHSInputShape { 1,1,2,2 };
        std::vector<int32_t> outputShape   { 1,1,2,2 };

        std::vector<float> LHSInputValues = { 1, 2,
                                              3, 4 };

        std::vector<float> RHSInputValues = { 5, 6,
                                              7, 8  };

        std::vector<float> expectedOutputValues = { 19, 22,
                                                    43, 50 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_FLOAT32,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               false,
                               false);
    }

    void BatchMatMul4DInt8SimpleTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 1,1,2,2};
        std::vector<int32_t> RHSInputShape { 1,1,2,2 };
        std::vector<int32_t> outputShape   { 1,1,2,2 };

        std::vector<int8_t> LHSInputValues = { 1, 2,
                                              3, 4 };

        std::vector<int8_t> RHSInputValues = { 5, 6,
                                              7, 8 };

        std::vector<int8_t> expectedOutputValues = { 19, 22,
                                                    43, 50 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_INT8,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               false,
                               false);
    }

    void BatchMatMul3DFp32BatchTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 2,2,2 };
        std::vector<int32_t> RHSInputShape { 2,2,2 };
        std::vector<int32_t> outputShape   { 2,2,2 };

        std::vector<float> LHSInputValues = { 1, 2,
                                              3, 4,

                                              9, 10,
                                              11, 12 };

        std::vector<float> RHSInputValues = { 5, 6,
                                              7, 8,

                                              13, 14,
                                              15, 16 };

        std::vector<float> expectedOutputValues = { 19, 22,
                                                    43, 50,

                                                    267, 286,
                                                    323, 346 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_FLOAT32,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               false,
                               false);
    }

    void BatchMatMul3DInt8BatchTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 2,2,2 };
        std::vector<int32_t> RHSInputShape { 2,2,2 };
        std::vector<int32_t> outputShape   { 2,2,2 };

        std::vector<int8_t> LHSInputValues = { 1, 2,
                                              3, 4,

                                              9, 10,
                                              11, 12 };

        std::vector<int8_t> RHSInputValues = { 5, 6,
                                              7, 8,

                                              1, 2,
                                              3, 4 };

        std::vector<int8_t> expectedOutputValues = { 19, 22,
                                                    43, 50,

                                                    39, 58,
                                                    47, 70 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_INT8,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               false,
                               false);
    }

    void BatchMatMul3DFp32BroadcastTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 2,2,2 };
        std::vector<int32_t> RHSInputShape { 2,2 };
        std::vector<int32_t> outputShape   { 2,2,2 };

        std::vector<float> LHSInputValues = { 1, 2,
                                              3, 4,

                                              9, 10,
                                              11, 12 };

        std::vector<float> RHSInputValues = { 13, 14,
                                              15, 16 };

        std::vector<float> expectedOutputValues = {  43, 46,
                                                     99, 106,

                                                     267, 286,
                                                     323, 346 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_FLOAT32,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               false,
                               false);
    }

    void BatchMatMul3DInt8BroadcastTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 2,2,2 };
        std::vector<int32_t> RHSInputShape { 1,2,2 };
        std::vector<int32_t> outputShape   { 2,2,2 };

        std::vector<int8_t> LHSInputValues = { 1, 2,
                                              3, 4,

                                              9, 10,
                                              11, 12 };

        std::vector<int8_t> RHSInputValues = { 1, 2,
                                               3, 4 };

        std::vector<int8_t> expectedOutputValues = {  7,  10,
                                                      15, 22,

                                                      39, 58,
                                                      47, 70 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_INT8,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               false,
                               false);
    }

    void BatchMatMul3D2DFp32BroadcastTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 2,2,2 };
        std::vector<int32_t> RHSInputShape { 2,2 };
        std::vector<int32_t> outputShape   { 2,2,2 };

        std::vector<float> LHSInputValues = { 1, 2,
                                              3, 4,

                                              9, 10,
                                              11, 12 };

        std::vector<float> RHSInputValues = { 13, 14,
                                              15, 16 };

        std::vector<float> expectedOutputValues = {  43, 46,
                                                     99, 106,

                                                     267, 286,
                                                     323, 346 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_FLOAT32,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               false,
                               false);
    }

    void BatchMatMul3D2DInt8BroadcastTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 2,2,2 };
        std::vector<int32_t> RHSInputShape { 2,2 };
        std::vector<int32_t> outputShape   { 2,2,2 };

        std::vector<int8_t> LHSInputValues = { 1, 2,
                                              3, 4,

                                              9, 10,
                                              11, 12 };

        std::vector<int8_t> RHSInputValues = { 1, 2,
                                               3, 4 };

        std::vector<int8_t> expectedOutputValues = {  7, 10,
                                                      15, 22,

                                                      39, 58,
                                                      47, 70 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_INT8,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               false,
                               false);
    }

    void BatchMatMul2DFp32TinyTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 1,1 };
        std::vector<int32_t> RHSInputShape { 1,1 };
        std::vector<int32_t> outputShape   { 1,1 };

        std::vector<float> LHSInputValues = { 3 };

        std::vector<float> RHSInputValues = { 5 };

        std::vector<float> expectedOutputValues = { 15 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_FLOAT32,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               false,
                               false);
    }
    void BatchMatMul2DInt8TinyTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 1,1 };
        std::vector<int32_t> RHSInputShape { 1,1 };
        std::vector<int32_t> outputShape   { 1,1 };

        std::vector<int8_t> LHSInputValues = { 3 };

        std::vector<int8_t> RHSInputValues = { 5 };

        std::vector<int8_t> expectedOutputValues = { 15 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL,
                                ::tflite::TensorType_INT8,
                                backends,
                                LHSInputShape,
                                RHSInputShape,
                                outputShape,
                                LHSInputValues,
                                RHSInputValues,
                                expectedOutputValues,
                                false,
                                false);
    }

    void BatchMatMulNonSquareFp32Test(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 2,5,3 };
        std::vector<int32_t> RHSInputShape { 2,3,4 };
        std::vector<int32_t> outputShape   { 2,5,4 };

        std::vector<float> LHSInputValues = { 8, 8, 4,
                                              6, 1, 3,
                                              8, 8, 3,
                                              8, 9, 8,
                                              5, 4, 4,

                                              1, 8, 5,
                                              7, 1, 1,
                                              8, 7, 9,
                                              3, 2, 7,
                                              8, 5, 3 };

        std::vector<float> RHSInputValues = { 6, 2, 3, 2,
                                              6, 2, 2, 8,
                                              3, 7, 8, 1,

                                              7, 2, 9, 5,
                                              2, 3, 1, 3,
                                              2, 7, 7, 5 };

        std::vector<float> expectedOutputValues = { 108, 60, 72, 84,
                                                    51, 35, 44, 23,
                                                    105, 53, 64, 83,
                                                    126, 90, 106, 96,
                                                    66, 46, 55, 46,

                                                    33, 61, 52, 54,
                                                    53, 24, 71, 43,
                                                    88, 100, 142, 106,
                                                    39, 61, 78, 56,
                                                    72, 52, 98, 70 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_FLOAT32,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               false,
                               false);
    }

    void BatchMatMulNonSquareInt8Test(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 2,5,3 };
        std::vector<int32_t> RHSInputShape { 2,3,4 };
        std::vector<int32_t> outputShape   { 2,5,4 };

        std::vector<int8_t> LHSInputValues = { 8, 8, 4,
                                              6, 1, 3,
                                              8, 8, 3,
                                              8, 9, 8,
                                              5, 4, 4,

                                              1, 8, 5,
                                              7, 1, 1,
                                              8, 7, 9,
                                              3, 2, 7,
                                              8, 5, 3 };

        std::vector<int8_t> RHSInputValues = { 6, 2, 3, 2,
                                              6, 2, 2, 8,
                                              3, 7, 8, 1,

                                              7, 2, 3, 5,
                                              2, 3, 1, 3,
                                              2, 7, 7, 5 };

        std::vector<int8_t> expectedOutputValues = { 108, 60, 72, 84,
                                                    51, 35, 44, 23,
                                                    105, 53, 64, 83,
                                                    126, 90, 106, 96,
                                                    66, 46, 55, 46,

                                                    33, 61, 46, 54,
                                                    53, 24, 29, 43,
                                                    88, 100, 94, 106,
                                                    39, 61, 60, 56,
                                                    72, 52, 50, 70 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_INT8,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               false,
                               false);
    }

    void BatchMatMul2DFp32SimpleAdjointTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 3,3 };
        std::vector<int32_t> RHSInputShape { 3,3 };
        std::vector<int32_t> outputShape   { 3,3 };

        std::vector<float> LHSInputValues = { 3, 1, 1,
                                              1, 3, -1,
                                              2, 4, 1 };

        std::vector<float> RHSInputValues = { 1, 0, 0,
                                              0, 1, 0,
                                              0, 0, 1 };

        std::vector<float> expectedOutputValues = { 3, 1, 2,
                                                    1, 3, 4,
                                                    1, -1, 1 };

        BatchMatMulTest<float>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_FLOAT32,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               true,
                               false);
    }

    void BatchMatMul2DInt8SimpleAdjointTest(std::vector<armnn::BackendId>& backends)
    {
        // Set input data
        std::vector<int32_t> LHSInputShape { 3,3 };
        std::vector<int32_t> RHSInputShape { 3,3 };
        std::vector<int32_t> outputShape   { 3,3 };

        std::vector<int8_t> LHSInputValues = { 3, 1, 1,
                                              1, 3, -1,
                                              2, 4, 1 };

        std::vector<int8_t> RHSInputValues = { 1, 0, 0,
                                              0, 1, 0,
                                              0, 0, 1 };

        std::vector<int8_t> expectedOutputValues = { 3, 1, 2,
                                                     1, 3, 4,
                                                     1, -1, 1 };

        BatchMatMulTest<int8_t>(tflite::BuiltinOperator_BATCH_MATMUL,
                               ::tflite::TensorType_INT8,
                               backends,
                               LHSInputShape,
                               RHSInputShape,
                               outputShape,
                               LHSInputValues,
                               RHSInputValues,
                               expectedOutputValues,
                               true,
                               false);
    }

    TEST_SUITE("BATCH_MATMUL_CpuRefTests")
    {
        TEST_CASE("BATCH_MATMUL_Fp32_CpuRefTests")
        {
            std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
            BatchMatMul2DFp32SimpleTest       (backends);
            BatchMatMul3DFp32SimpleTest       (backends);
            BatchMatMul4DFp32SimpleTest       (backends);
            BatchMatMul3DFp32BatchTest        (backends);
            BatchMatMul3DFp32BroadcastTest    (backends);
            BatchMatMul3D2DFp32BroadcastTest  (backends);
            BatchMatMul2DFp32TinyTest         (backends);
            BatchMatMulNonSquareFp32Test      (backends);
            BatchMatMul2DFp32SimpleAdjointTest(backends);
        }

        TEST_CASE("BATCH_MATMUL_Int8_CpuRefTests")
        {
            std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
            BatchMatMul2DInt8SimpleTest       (backends);
            BatchMatMul3DInt8SimpleTest       (backends);
            BatchMatMul4DInt8SimpleTest       (backends);
            BatchMatMul3DInt8BatchTest        (backends);
            BatchMatMul3DInt8BroadcastTest    (backends);
            BatchMatMul3D2DInt8BroadcastTest  (backends);
            BatchMatMul2DInt8TinyTest         (backends);
            BatchMatMulNonSquareInt8Test      (backends);
            BatchMatMul2DInt8SimpleAdjointTest(backends);
        }
    }

    TEST_SUITE("BATCH_MATMUL_CpuAccTests")
    {
        TEST_CASE("BATCH_MATMUL_Fp32_CpuAccTests")
        {
            std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
            BatchMatMul2DFp32SimpleTest       (backends);
            BatchMatMul3DFp32SimpleTest       (backends);
            BatchMatMul4DFp32SimpleTest       (backends);
            BatchMatMul3DFp32BatchTest        (backends);
            BatchMatMul3DFp32BroadcastTest    (backends);
            BatchMatMul3D2DFp32BroadcastTest  (backends);
            BatchMatMul2DFp32TinyTest         (backends);
            BatchMatMulNonSquareFp32Test      (backends);
            BatchMatMul2DFp32SimpleAdjointTest(backends);
        }
    }
    TEST_SUITE("BATCH_MATMUL_GpuAccTests")
    {
        TEST_CASE("BATCH_MATMUL_Fp32_GpuAccTests")
        {
            std::vector <armnn::BackendId> backends = {armnn::Compute::GpuAcc};
            BatchMatMul2DFp32SimpleTest       (backends);
            BatchMatMul3DFp32SimpleTest       (backends);
            BatchMatMul4DFp32SimpleTest       (backends);
            BatchMatMul3DFp32BatchTest        (backends);
            BatchMatMul3DFp32BroadcastTest    (backends);
            BatchMatMul3D2DFp32BroadcastTest  (backends);
            BatchMatMul2DFp32TinyTest         (backends);
            BatchMatMulNonSquareFp32Test      (backends);
            BatchMatMul2DFp32SimpleAdjointTest(backends);
        }
    }
}
