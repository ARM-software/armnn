//
// Copyright © 2021, 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReduceTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

void ReduceUint8KeepDimsTest(tflite::BuiltinOperator reduceOperatorCode,
                             std::vector<uint8_t>& expectedOutputValues,
                             const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> input0Shape { 1, 1, 2, 3 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 1, 1, 1, 3 };

    std::vector<uint8_t> input0Values { 1, 2, 3,
                                        4, 3, 1  }; // Inputs
    std::vector<int32_t> input1Values { 2 }; // Axis

    ReduceTest<uint8_t>(reduceOperatorCode,
                        ::tflite::TensorType_UINT8,
                        input0Shape,
                        input1Shape,
                        expectedOutputShape,
                        input0Values,
                        input1Values,
                        expectedOutputValues,
                        true,
                        backends);
}

void ReduceUint8Test(tflite::BuiltinOperator reduceOperatorCode,
                     std::vector<uint8_t>& expectedOutputValues,
                     const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> input0Shape { 1, 1, 2, 3 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 1, 1, 3 };

    std::vector<uint8_t> input0Values { 1, 2, 3,
                                        4, 3, 1 }; // Inputs
    std::vector<int32_t> input1Values { 2 }; // Axis

    ReduceTest<uint8_t>(reduceOperatorCode,
                        ::tflite::TensorType_UINT8,
                        input0Shape,
                        input1Shape,
                        expectedOutputShape,
                        input0Values,
                        input1Values,
                        expectedOutputValues,
                        false,
                        backends);
}

void ReduceFp32KeepDimsTest(tflite::BuiltinOperator reduceOperatorCode,
                            std::vector<float>& expectedOutputValues,
                            const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> input0Shape { 1, 1, 2, 3 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 1, 1, 1, 3 };

    std::vector<float>   input0Values { 1001.0f, 11.0f,   1003.0f,
                                        10.0f,   1002.0f, 12.0f }; // Inputs
    std::vector<int32_t> input1Values { 2 }; // Axis

    ReduceTest<float>(reduceOperatorCode,
                      ::tflite::TensorType_FLOAT32,
                      input0Shape,
                      input1Shape,
                      expectedOutputShape,
                      input0Values,
                      input1Values,
                      expectedOutputValues,
                      true,
                      backends);
}

void ReduceFp32Test(tflite::BuiltinOperator reduceOperatorCode,
                    std::vector<float>& expectedOutputValues,
                    const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> input0Shape { 1, 1, 2, 3 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 1, 1, 3 };

    std::vector<float>   input0Values { 1001.0f, 11.0f,   1003.0f,
                                        10.0f,   1002.0f, 12.0f }; // Inputs
    std::vector<int32_t> input1Values { 2 }; // Axis

    ReduceTest<float>(reduceOperatorCode,
                      ::tflite::TensorType_FLOAT32,
                      input0Shape,
                      input1Shape,
                      expectedOutputShape,
                      input0Values,
                      input1Values,
                      expectedOutputValues,
                      false,
                      backends);
}

// REDUCE_MAX Tests
TEST_SUITE("ReduceMaxTests")
{

TEST_CASE ("ReduceMax_Uint8_KeepDims_Test")
{
    std::vector<uint8_t> expectedOutputValues { 4, 3, 3 };
    ReduceUint8KeepDimsTest(tflite::BuiltinOperator_REDUCE_MAX,
                            expectedOutputValues);
}

TEST_CASE ("ReduceMax_Uint8_Test")
{
    std::vector<uint8_t> expectedOutputValues { 4, 3, 3 };
    ReduceUint8Test(tflite::BuiltinOperator_REDUCE_MAX,
                    expectedOutputValues);
}

TEST_CASE ("ReduceMax_Fp32_KeepDims_Test")
{
    std::vector<float>   expectedOutputValues { 1001.0f, 1002.0f, 1003.0f };
    ReduceFp32KeepDimsTest(tflite::BuiltinOperator_REDUCE_MAX,
                           expectedOutputValues);
}

TEST_CASE ("ReduceMax_Fp32_Test")
{
    std::vector<float>   expectedOutputValues { 1001.0f, 1002.0f, 1003.0f };
    ReduceFp32Test(tflite::BuiltinOperator_REDUCE_MAX,
                   expectedOutputValues);
}

} // End of ReduceMaxTests

// REDUCE_MIN Tests
TEST_SUITE("ReduceMinTests")
{

TEST_CASE ("ReduceMin_Fp32_Test")
{
    std::vector<float>   expectedOutputValues { 10.0f, 11.0f, 12.0f };
    ReduceFp32Test(tflite::BuiltinOperator_REDUCE_MIN,
                   expectedOutputValues);
}

} // End of ReduceMinTests

// SUM Tests
TEST_SUITE("SumTests")
{

TEST_CASE ("Sum_Uint8_KeepDims_Test")
{
    std::vector<uint8_t> expectedOutputValues { 5, 5, 4 };
    ReduceUint8KeepDimsTest(tflite::BuiltinOperator_SUM,
                            expectedOutputValues);
}

TEST_CASE ("Sum_Fp32_Test")
{
    std::vector<float>   expectedOutputValues { 1011.0f, 1013.0f, 1015.0f };
    ReduceFp32Test(tflite::BuiltinOperator_SUM,
                   expectedOutputValues);
}

} // End of SumTests

// PROD Tests
TEST_SUITE("ProdTests")
{

TEST_CASE ("Prod_Uint8_KeepDims_Test")
{
    std::vector<uint8_t> expectedOutputValues { 4, 6, 3 };
    ReduceUint8KeepDimsTest(tflite::BuiltinOperator_REDUCE_PROD,
                            expectedOutputValues);
}

TEST_CASE ("Prod_Fp32_Test")
{
    std::vector<float>   expectedOutputValues { 10010.0f, 11022.0f, 12036.0f };
    ReduceFp32Test(tflite::BuiltinOperator_REDUCE_PROD,
                   expectedOutputValues);
}

} // End of ProdTests

} // namespace armnnDelegate