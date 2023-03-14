//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchSpaceTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

// BatchToSpaceND Operator
void BatchToSpaceNDFp32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 4, 1, 1, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<float> inputValues { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<float> expectedOutputValues { 1.0f, 2.0f, 3.0f, 4.0f };

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    BatchSpaceTest<float>(tflite::BuiltinOperator_BATCH_TO_SPACE_ND,
                          ::tflite::TensorType_FLOAT32,
                          backends,
                          inputShape,
                          expectedOutputShape,
                          inputValues,
                          blockShape,
                          crops,
                          expectedOutputValues);
}

void BatchToSpaceNDFp32BatchOneTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<float> inputValues { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<float> expectedOutputValues { 1.0f, 2.0f, 3.0f, 4.0f };

    std::vector<unsigned int> blockShape({1, 1});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    BatchSpaceTest<float>(tflite::BuiltinOperator_BATCH_TO_SPACE_ND,
                          ::tflite::TensorType_FLOAT32,
                          backends,
                          inputShape,
                          expectedOutputShape,
                          inputValues,
                          blockShape,
                          crops,
                          expectedOutputValues);
}

void BatchToSpaceNDUint8Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 4, 1, 1, 3 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 3 };

    std::vector<uint8_t> inputValues { 1, 2, 3, 4, 5, 6, 7 };
    std::vector<uint8_t> expectedOutputValues { 1, 2, 3, 4, 5, 6, 7 };

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    BatchSpaceTest<uint8_t>(tflite::BuiltinOperator_BATCH_TO_SPACE_ND,
                          ::tflite::TensorType_UINT8,
                          backends,
                          inputShape,
                          expectedOutputShape,
                          inputValues,
                          blockShape,
                          crops,
                          expectedOutputValues);
}

// SpaceToBatchND Operator
void SpaceToBatchNDFp32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 4, 1, 1, 1 };

    std::vector<float> inputValues { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<float> expectedOutputValues { 1.0f, 2.0f, 3.0f, 4.0f };

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> padding = {{0, 0}, {0, 0}};

    BatchSpaceTest<float>(tflite::BuiltinOperator_SPACE_TO_BATCH_ND,
                          ::tflite::TensorType_FLOAT32,
                          backends,
                          inputShape,
                          expectedOutputShape,
                          inputValues,
                          blockShape,
                          padding,
                          expectedOutputValues);
}

void SpaceToBatchNDFp32PaddingTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 2, 2, 4, 1 };
    std::vector<int32_t> expectedOutputShape { 8, 1, 3, 1 };

    std::vector<float> inputValues { 1.0f,  2.0f,  3.0f,  4.0f,
                                     5.0f,  6.0f,  7.0f,  8.0f,
                                     9.0f,  10.0f, 11.0f, 12.0f,
                                     13.0f, 14.0f, 15.0f, 16.0f };

    std::vector<float> expectedOutputValues { 0.0f, 1.0f, 3.0f,  0.0f, 9.0f, 11.0f,
                                              0.0f, 2.0f, 4.0f,  0.0f, 10.0f, 12.0f,
                                              0.0f, 5.0f, 7.0f,  0.0f, 13.0f, 15.0f,
                                              0.0f, 6.0f, 8.0f,  0.0f, 14.0f, 16.0f };

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> padding = {{0, 0}, {2, 0}};

    BatchSpaceTest<float>(tflite::BuiltinOperator_SPACE_TO_BATCH_ND,
                          ::tflite::TensorType_FLOAT32,
                          backends,
                          inputShape,
                          expectedOutputShape,
                          inputValues,
                          blockShape,
                          padding,
                          expectedOutputValues);
}

void SpaceToBatchNDUint8Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 1, 2, 2, 3 };
    std::vector<int32_t> expectedOutputShape { 4, 1, 1, 3 };

    std::vector<uint8_t> inputValues { 1, 2, 3, 4, 5, 6, 7 };
    std::vector<uint8_t> expectedOutputValues { 1, 2, 3, 4, 5, 6, 7 };

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> padding = {{0, 0}, {0, 0}};

    BatchSpaceTest<uint8_t>(tflite::BuiltinOperator_SPACE_TO_BATCH_ND,
                            ::tflite::TensorType_UINT8,
                            backends,
                            inputShape,
                            expectedOutputShape,
                            inputValues,
                            blockShape,
                            padding,
                            expectedOutputValues);
}

// BatchToSpaceND Tests
TEST_SUITE("BatchToSpaceND_CpuAccTests")
{

TEST_CASE ("BatchToSpaceND_Fp32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    BatchToSpaceNDFp32Test(backends);
}

TEST_CASE ("BatchToSpaceND_Fp32_BatchOne_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    BatchToSpaceNDFp32BatchOneTest(backends);
}

TEST_CASE ("BatchToSpaceND_Uint8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    BatchToSpaceNDUint8Test(backends);
}

}

TEST_SUITE("BatchToSpaceND_GpuAccTests")
{

TEST_CASE ("BatchToSpaceND_Fp32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    BatchToSpaceNDFp32Test(backends);
}

TEST_CASE ("BatchToSpaceND_Fp32_BatchOne_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    BatchToSpaceNDFp32BatchOneTest(backends);
}

TEST_CASE ("BatchToSpaceND_Uint8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    BatchToSpaceNDUint8Test(backends);
}

}

TEST_SUITE("BatchToSpaceND_CpuRefTests")
{

TEST_CASE ("BatchToSpaceND_Fp32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    BatchToSpaceNDFp32Test(backends);
}

TEST_CASE ("BatchToSpaceND_Fp32_BatchOne_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    BatchToSpaceNDFp32BatchOneTest(backends);
}

TEST_CASE ("BatchToSpaceND_Uint8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    BatchToSpaceNDUint8Test(backends);
}

}

// SpaceToBatchND Tests
TEST_SUITE("SpaceToBatchND_CpuAccTests")
{

TEST_CASE ("SpaceToBatchND_Fp32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    SpaceToBatchNDFp32Test(backends);
}

TEST_CASE ("SpaceToBatchND_Fp32_Padding_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    SpaceToBatchNDFp32PaddingTest(backends);
}

TEST_CASE ("SpaceToBatchND_Uint8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    SpaceToBatchNDUint8Test(backends);
}

}

TEST_SUITE("SpaceToBatchND_GpuAccTests")
{

TEST_CASE ("SpaceToBatchND_Fp32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    SpaceToBatchNDFp32Test(backends);
}

TEST_CASE ("SpaceToBatchND_Fp32_Padding_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    SpaceToBatchNDFp32PaddingTest(backends);
}

TEST_CASE ("SpaceToBatchND_Uint8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    SpaceToBatchNDUint8Test(backends);
}

}

TEST_SUITE("SpaceToBatchND_CpuRefTests")
{

TEST_CASE ("SpaceToBatchND_Fp32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    SpaceToBatchNDFp32Test(backends);
}

TEST_CASE ("SpaceToBatchND_Fp32_Padding_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    SpaceToBatchNDFp32PaddingTest(backends);
}

TEST_CASE ("SpaceToBatchND_Uint8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    SpaceToBatchNDUint8Test(backends);
}

}

} // namespace armnnDelegate