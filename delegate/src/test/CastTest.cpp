//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CastTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void CastUint8ToFp32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape  {1, 3, 2, 3};

    std::vector<uint8_t> inputValues { 1, 3, 1, 3, 1, 3, 1, 3, 1,
                                        3, 1, 3, 1, 2, 1, 3, 1, 3 };

    std::vector<float> expectedOutputValues { 1.0f, 3.0f, 1.0f, 3.0f, 1.0f, 3.0f, 1.0f, 3.0f, 1.0f,
                                              3.0f, 1.0f, 3.0f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 3.0f };

    CastTest<uint8_t, float>(::tflite::TensorType_UINT8,
                             ::tflite::TensorType_FLOAT32,
                             backends,
                             inputShape,
                             inputValues,
                             expectedOutputValues);
}

void CastInt32ToFp32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape  {1, 3, 2, 3};

    std::vector<int32_t> inputValues { -1, -3, -1, -3, -1, -3, -1, -3, 1,
                                       3, 1, 3, 1, 2, 1, 3, 1, 3 };

    std::vector<float> expectedOutputValues { -1.0f, -3.0f, -1.0f, -3.0f, -1.0f, -3.0f, -1.0f, -3.0f, 1.0f,
                                              3.0f, 1.0f, 3.0f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 3.0f };

    CastTest<int32_t, float>(::tflite::TensorType_INT32,
                             ::tflite::TensorType_FLOAT32,
                             backends,
                             inputShape,
                             inputValues,
                             expectedOutputValues);
}

// CAST Test Suite
TEST_SUITE("CAST_CpuRefTests")
{

TEST_CASE ("CAST_UINT8_TO_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    CastUint8ToFp32Test(backends);
}

TEST_CASE ("CAST_INT32_TO_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    CastInt32ToFp32Test(backends);
}

}

TEST_SUITE("CAST_CpuAccTests")
{

TEST_CASE ("CAST_INT32_TO_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    CastInt32ToFp32Test(backends);
}

}

TEST_SUITE("CAST_GpuAccTests")
{

TEST_CASE ("CAST_INT32_TO_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    CastInt32ToFp32Test(backends);
}

}
// End of CAST Test Suite

} // namespace armnnDelegate