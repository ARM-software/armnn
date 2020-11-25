//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatherTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

// GATHER Operator
void GatherUint8Test(std::vector<armnn::BackendId>& backends)
{

    std::vector<int32_t> paramsShape{8};
    std::vector<int32_t> indicesShape{3};
    std::vector<int32_t> expectedOutputShape{3};

    int32_t              axis = 0;
    std::vector<uint8_t> paramsValues{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int32_t> indicesValues{7, 6, 5};
    std::vector<uint8_t> expectedOutputValues{8, 7, 6};

    GatherTest<uint8_t>(::tflite::TensorType_UINT8,
                        backends,
                        paramsShape,
                        indicesShape,
                        expectedOutputShape,
                        axis,
                        paramsValues,
                        indicesValues,
                        expectedOutputValues);
}

void GatherFp32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> paramsShape{8};
    std::vector<int32_t> indicesShape{3};
    std::vector<int32_t> expectedOutputShape{3};

    int32_t              axis = 0;
    std::vector<float>   paramsValues{1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f};
    std::vector<int32_t> indicesValues{7, 6, 5};
    std::vector<float>   expectedOutputValues{8.8f, 7.7f, 6.6f};

    GatherTest<float>(::tflite::TensorType_FLOAT32,
                      backends,
                      paramsShape,
                      indicesShape,
                      expectedOutputShape,
                      axis,
                      paramsValues,
                      indicesValues,
                      expectedOutputValues);
}

// GATHER Test Suite
TEST_SUITE("GATHER_CpuRefTests")
{

TEST_CASE ("GATHER_Uint8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    GatherUint8Test(backends);
}

TEST_CASE ("GATHER_Fp32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    GatherFp32Test(backends);
}

}

TEST_SUITE("GATHER_CpuAccTests")
{

TEST_CASE ("GATHER_Uint8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    GatherUint8Test(backends);
}

TEST_CASE ("GATHER_Fp32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    GatherFp32Test(backends);
}

}

TEST_SUITE("GATHER_GpuAccTests")
{

TEST_CASE ("GATHER_Uint8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    GatherUint8Test(backends);
}

TEST_CASE ("GATHER_Fp32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    GatherFp32Test(backends);
}

}
// End of GATHER Test Suite

} // namespace armnnDelegate