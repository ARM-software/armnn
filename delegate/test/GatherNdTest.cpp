//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatherNdTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

// GATHER_ND Operator
void GatherNdUint8Test(std::vector<armnn::BackendId>& backends)
{

    std::vector<int32_t> paramsShape{ 5, 2 };
    std::vector<int32_t> indicesShape{ 3, 1 };
    std::vector<int32_t> expectedOutputShape{ 3, 2 };

    std::vector<uint8_t> paramsValues{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    std::vector<int32_t> indicesValues{ 1, 0, 4 };
    std::vector<uint8_t> expectedOutputValues{ 3, 4, 1, 2, 9, 10 };

    GatherNdTest<uint8_t>(::tflite::TensorType_UINT8,
                          backends,
                          paramsShape,
                          indicesShape,
                          expectedOutputShape,
                          paramsValues,
                          indicesValues,
                          expectedOutputValues);
}

void GatherNdFp32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> paramsShape{ 5, 2 };
    std::vector<int32_t> indicesShape{ 3, 1 };
    std::vector<int32_t> expectedOutputShape{ 3, 2 };

    std::vector<float>   paramsValues{ 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f, 10.10f };
    std::vector<int32_t> indicesValues{ 1, 0, 4 };
    std::vector<float>   expectedOutputValues{ 3.3f, 4.4f, 1.1f, 2.2f, 9.9f, 10.10f };

    GatherNdTest<float>(::tflite::TensorType_FLOAT32,
                        backends,
                        paramsShape,
                        indicesShape,
                        expectedOutputShape,
                        paramsValues,
                        indicesValues,
                        expectedOutputValues);
}

// GATHER_ND Test Suite
TEST_SUITE("GATHER_ND_CpuRefTests")
{

TEST_CASE ("GATHER_ND_Uint8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    GatherNdUint8Test(backends);
}

TEST_CASE ("GATHER_ND_Fp32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    GatherNdFp32Test(backends);
}

}

TEST_SUITE("GATHER_ND_CpuAccTests")
{

TEST_CASE ("GATHER_ND_Uint8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    GatherNdUint8Test(backends);
}

TEST_CASE ("GATHER_ND_Fp32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    GatherNdFp32Test(backends);
}

}

TEST_SUITE("GATHER_ND_GpuAccTests")
{

TEST_CASE ("GATHER_ND_Uint8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    GatherNdUint8Test(backends);
}

TEST_CASE ("GATHER_ND_Fp32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    GatherNdFp32Test(backends);
}

}
// End of GATHER_ND Test Suite

} // namespace armnnDelegate