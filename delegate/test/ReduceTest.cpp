//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReduceTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void ReduceUint8KeepDimsTest(tflite::BuiltinOperator reduceOperatorCode,
                             std::vector<armnn::BackendId>& backends,
                             std::vector<uint8_t>& expectedOutputValues)
{
    std::vector<int32_t> input0Shape { 1, 1, 2, 3 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 1, 1, 1, 3 };

    std::vector<uint8_t> input0Values { 1, 2, 3,
                                        4, 3, 1  }; // Inputs
    std::vector<int32_t> input1Values { 2 }; // Axis

    ReduceTest<uint8_t>(reduceOperatorCode,
                        ::tflite::TensorType_UINT8,
                        backends,
                        input0Shape,
                        input1Shape,
                        expectedOutputShape,
                        input0Values,
                        input1Values,
                        expectedOutputValues,
                        true);
}

void ReduceUint8Test(tflite::BuiltinOperator reduceOperatorCode,
                     std::vector<armnn::BackendId>& backends,
                     std::vector<uint8_t>& expectedOutputValues)
{
    std::vector<int32_t> input0Shape { 1, 1, 2, 3 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 1, 1, 3 };

    std::vector<uint8_t> input0Values { 1, 2, 3,
                                        4, 3, 1 }; // Inputs
    std::vector<int32_t> input1Values { 2 }; // Axis

    ReduceTest<uint8_t>(reduceOperatorCode,
                        ::tflite::TensorType_UINT8,
                        backends,
                        input0Shape,
                        input1Shape,
                        expectedOutputShape,
                        input0Values,
                        input1Values,
                        expectedOutputValues,
                        false);
}

void ReduceFp32KeepDimsTest(tflite::BuiltinOperator reduceOperatorCode,
                            std::vector<armnn::BackendId>& backends,
                            std::vector<float>& expectedOutputValues)
{
    std::vector<int32_t> input0Shape { 1, 1, 2, 3 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 1, 1, 1, 3 };

    std::vector<float>   input0Values { 1001.0f, 11.0f,   1003.0f,
                                        10.0f,   1002.0f, 12.0f }; // Inputs
    std::vector<int32_t> input1Values { 2 }; // Axis

    ReduceTest<float>(reduceOperatorCode,
                      ::tflite::TensorType_FLOAT32,
                      backends,
                      input0Shape,
                      input1Shape,
                      expectedOutputShape,
                      input0Values,
                      input1Values,
                      expectedOutputValues,
                      true);
}

void ReduceFp32Test(tflite::BuiltinOperator reduceOperatorCode,
                    std::vector<armnn::BackendId>& backends,
                    std::vector<float>& expectedOutputValues)
{
    std::vector<int32_t> input0Shape { 1, 1, 2, 3 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 1, 1, 3 };

    std::vector<float>   input0Values { 1001.0f, 11.0f,   1003.0f,
                                        10.0f,   1002.0f, 12.0f }; // Inputs
    std::vector<int32_t> input1Values { 2 }; // Axis

    ReduceTest<float>(reduceOperatorCode,
                      ::tflite::TensorType_FLOAT32,
                      backends,
                      input0Shape,
                      input1Shape,
                      expectedOutputShape,
                      input0Values,
                      input1Values,
                      expectedOutputValues,
                      false);
}

// REDUCE_MAX Tests
TEST_SUITE("ReduceMax_CpuRefTests")
{

TEST_CASE ("ReduceMax_Uint8_KeepDims_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    std::vector<uint8_t> expectedOutputValues { 4, 3, 3 };
    ReduceUint8KeepDimsTest(tflite::BuiltinOperator_REDUCE_MAX,
                            backends,
                            expectedOutputValues);
}

TEST_CASE ("ReduceMax_Uint8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    std::vector<uint8_t> expectedOutputValues { 4, 3, 3 };
    ReduceUint8Test(tflite::BuiltinOperator_REDUCE_MAX,
                    backends,
                    expectedOutputValues);
}

TEST_CASE ("ReduceMax_Fp32_KeepDims_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    std::vector<float>   expectedOutputValues { 1001.0f, 1002.0f, 1003.0f };
    ReduceFp32KeepDimsTest(tflite::BuiltinOperator_REDUCE_MAX,
                           backends,
                           expectedOutputValues);
}

TEST_CASE ("ReduceMax_Fp32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    std::vector<float>   expectedOutputValues { 1001.0f, 1002.0f, 1003.0f };
    ReduceFp32Test(tflite::BuiltinOperator_REDUCE_MAX,
                   backends,
                   expectedOutputValues);
}

} // End of ReduceMax_CpuRefTests

TEST_SUITE("ReduceMax_CpuAccTests")
{

TEST_CASE ("ReduceMax_Uint8_KeepDims_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    std::vector<uint8_t> expectedOutputValues { 4, 3, 3 };
    ReduceUint8KeepDimsTest(tflite::BuiltinOperator_REDUCE_MAX,
                            backends,
                            expectedOutputValues);
}

TEST_CASE ("ReduceMax_Uint8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    std::vector<uint8_t> expectedOutputValues { 4, 3, 3 };
    ReduceUint8Test(tflite::BuiltinOperator_REDUCE_MAX,
                    backends,
                    expectedOutputValues);
}


TEST_CASE ("ReduceMax_Fp32_KeepDims_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    std::vector<float>   expectedOutputValues { 1001.0f, 1002.0f, 1003.0f };
    ReduceFp32KeepDimsTest(tflite::BuiltinOperator_REDUCE_MAX,
                           backends,
                           expectedOutputValues);
}

TEST_CASE ("ReduceMax_Fp32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    std::vector<float>   expectedOutputValues { 1001.0f, 1002.0f, 1003.0f };
    ReduceFp32Test(tflite::BuiltinOperator_REDUCE_MAX,
                   backends,
                   expectedOutputValues);
}

} // End of ReduceMax_CpuAccTests

TEST_SUITE("ReduceMax_GpuAccTests")
{

TEST_CASE ("ReduceMax_Uint8_KeepDims_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    std::vector<uint8_t> expectedOutputValues { 4, 3, 3 };
    ReduceUint8KeepDimsTest(tflite::BuiltinOperator_REDUCE_MAX,
                            backends,
                            expectedOutputValues);
}

TEST_CASE ("ReduceMax_Uint8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    std::vector<uint8_t> expectedOutputValues { 4, 3, 3 };
    ReduceUint8Test(tflite::BuiltinOperator_REDUCE_MAX,
                    backends,
                    expectedOutputValues);
}


TEST_CASE ("ReduceMax_Fp32_KeepDims_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    std::vector<float>   expectedOutputValues { 1001.0f, 1002.0f, 1003.0f };
    ReduceFp32KeepDimsTest(tflite::BuiltinOperator_REDUCE_MAX,
                           backends,
                           expectedOutputValues);
}

TEST_CASE ("ReduceMax_Fp32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    std::vector<float>   expectedOutputValues { 1001.0f, 1002.0f, 1003.0f };
    ReduceFp32Test(tflite::BuiltinOperator_REDUCE_MAX,
                   backends,
                   expectedOutputValues);
}

} // End of ReduceMax_GpuAccTests

// REDUCE_MIN Tests
TEST_SUITE("ReduceMin_CpuRefTests")
{

TEST_CASE ("ReduceMin_Fp32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    std::vector<float>   expectedOutputValues { 10.0f, 11.0f, 12.0f };
    ReduceFp32Test(tflite::BuiltinOperator_REDUCE_MIN,
                   backends,
                   expectedOutputValues);
}

} // End of ReduceMin_CpuRefTests

TEST_SUITE("ReduceMin_CpuAccTests")
{

TEST_CASE ("ReduceMin_Fp32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    std::vector<float>   expectedOutputValues { 10.0f, 11.0f, 12.0f };
    ReduceFp32Test(tflite::BuiltinOperator_REDUCE_MIN,
                   backends,
                   expectedOutputValues);
}

} // End of ReduceMin_CpuAccTests

TEST_SUITE("ReduceMin_GpuAccTests")
{

TEST_CASE ("ReduceMin_Fp32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    std::vector<float>   expectedOutputValues { 10.0f, 11.0f, 12.0f };
    ReduceFp32Test(tflite::BuiltinOperator_REDUCE_MIN,
                   backends,
                   expectedOutputValues);
}

} // End of ReduceMin_GpuAccTests

// SUM Tests
TEST_SUITE("Sum_CpuRefTests")
{

TEST_CASE ("Sum_Uint8_KeepDims_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    std::vector<uint8_t> expectedOutputValues { 5, 5, 4 };
    ReduceUint8KeepDimsTest(tflite::BuiltinOperator_SUM,
                            backends,
                            expectedOutputValues);
}

TEST_CASE ("Sum_Fp32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    std::vector<float>   expectedOutputValues { 1011.0f, 1013.0f, 1015.0f };
    ReduceFp32Test(tflite::BuiltinOperator_SUM,
                   backends,
                   expectedOutputValues);
}

} // End of Sum_CpuRefTests

TEST_SUITE("Sum_CpuAccTests")
{

TEST_CASE ("Sum_Uint8_KeepDims_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    std::vector<uint8_t> expectedOutputValues { 5, 5, 4 };
    ReduceUint8KeepDimsTest(tflite::BuiltinOperator_SUM,
                            backends,
                            expectedOutputValues);
}

TEST_CASE ("Sum_Fp32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    std::vector<float>   expectedOutputValues { 1011.0f, 1013.0f, 1015.0f };
    ReduceFp32Test(tflite::BuiltinOperator_SUM,
                   backends,
                   expectedOutputValues);
}

} // End of Sum_CpuAccTests

TEST_SUITE("Sum_GpuAccTests")
{

TEST_CASE ("Sum_Uint8_KeepDims_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    std::vector<uint8_t> expectedOutputValues { 5, 5, 4 };
    ReduceUint8KeepDimsTest(tflite::BuiltinOperator_SUM,
                            backends,
                            expectedOutputValues);
}

TEST_CASE ("Sum_Fp32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    std::vector<float>   expectedOutputValues { 1011.0f, 1013.0f, 1015.0f };
    ReduceFp32Test(tflite::BuiltinOperator_SUM,
                   backends,
                   expectedOutputValues);
}

} // End of Sum_GpuAccTests

// PROD Tests
TEST_SUITE("Prod_CpuRefTests")
{

TEST_CASE ("Prod_Uint8_KeepDims_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    std::vector<uint8_t> expectedOutputValues { 4, 6, 3 };
    ReduceUint8KeepDimsTest(tflite::BuiltinOperator_REDUCE_PROD,
                            backends,
                            expectedOutputValues);
}

TEST_CASE ("Prod_Fp32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    std::vector<float>   expectedOutputValues { 10010.0f, 11022.0f, 12036.0f };
    ReduceFp32Test(tflite::BuiltinOperator_REDUCE_PROD,
                   backends,
                   expectedOutputValues);
}

} // End of Prod_CpuRefTests

TEST_SUITE("Prod_CpuAccTests")
{

TEST_CASE ("Prod_Uint8_KeepDims_CpuAcc_Test" )
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    std::vector<uint8_t> expectedOutputValues { 4, 6, 3 };
    ReduceUint8KeepDimsTest(tflite::BuiltinOperator_REDUCE_PROD,
                            backends,
                            expectedOutputValues);
}

TEST_CASE ("Prod_Fp32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    std::vector<float>   expectedOutputValues { 10010.0f, 11022.0f, 12036.0f };
    ReduceFp32Test(tflite::BuiltinOperator_REDUCE_PROD,
                   backends,
                   expectedOutputValues);
}

} // End of Prod_CpuAccTests

TEST_SUITE("Prod_GpuAccTests")
{

TEST_CASE ("Prod_Uint8_KeepDims_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    std::vector<uint8_t> expectedOutputValues { 4, 6, 3 };
    ReduceUint8KeepDimsTest(tflite::BuiltinOperator_REDUCE_PROD,
                            backends,
                            expectedOutputValues);
}

TEST_CASE ("Prod_Fp32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    std::vector<float>   expectedOutputValues { 10010.0f, 11022.0f, 12036.0f };
    ReduceFp32Test(tflite::BuiltinOperator_REDUCE_PROD,
                   backends,
                   expectedOutputValues);
}

} // End of Prod_GpuAccTests

} // namespace armnnDelegate