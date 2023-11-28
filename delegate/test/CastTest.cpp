//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CastTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>

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
                             inputShape,
                             inputValues,
                             expectedOutputValues,
                             1.0f,
                             0,
                             backends);
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
                             inputShape,
                             inputValues,
                             expectedOutputValues,
                             1.0f,
                             0,
                             backends);
}

// CAST Test Suite
TEST_SUITE("CASTTests")
{

TEST_CASE ("CAST_UINT8_TO_FP32_CpuRef_Test")
{
    // This only works on CpuRef.
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    CastUint8ToFp32Test(backends);
}

TEST_CASE ("CAST_INT32_TO_FP32_Test")
{
    std::vector<armnn::BackendId> backends = {};
    CastInt32ToFp32Test(backends);
}

}
// End of CAST Test Suite

} // namespace armnnDelegate