//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "UnpackTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

template <typename T>
void UnpackAxis0Num4Test(tflite::TensorType tensorType, std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 4, 1, 6 };
    std::vector<int32_t> expectedOutputShape { 1, 6 };

    std::vector<T> inputValues { 1, 2, 3, 4, 5, 6,
                                 7, 8, 9, 10, 11, 12,
                                 13, 14, 15, 16, 17, 18,
                                 19, 20, 21, 22, 23, 24 };

    std::vector<T> expectedOutputValues0 { 1, 2, 3, 4, 5, 6 };
    std::vector<T> expectedOutputValues1 { 7, 8, 9, 10, 11, 12 };
    std::vector<T> expectedOutputValues2 { 13, 14, 15, 16, 17, 18 };
    std::vector<T> expectedOutputValues3 { 19, 20, 21, 22, 23, 24 };

    std::vector<std::vector<T>> expectedOutputValues{ expectedOutputValues0,
                                                      expectedOutputValues1,
                                                      expectedOutputValues2,
                                                      expectedOutputValues3 };

    UnpackTest<T>(tflite::BuiltinOperator_UNPACK,
                  tensorType,
                  backends,
                  inputShape,
                  expectedOutputShape,
                  inputValues,
                  expectedOutputValues,
                  0);
}

template <typename T>
void UnpackAxis2Num6Test(tflite::TensorType tensorType, std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 4, 1, 6 };
    std::vector<int32_t> expectedOutputShape { 4, 1 };

    std::vector<T> inputValues { 1, 2, 3, 4, 5, 6,
                                 7, 8, 9, 10, 11, 12,
                                 13, 14, 15, 16, 17, 18,
                                 19, 20, 21, 22, 23, 24 };

    std::vector<T> expectedOutputValues0 { 1, 7, 13, 19 };
    std::vector<T> expectedOutputValues1 { 2, 8, 14, 20 };
    std::vector<T> expectedOutputValues2 { 3, 9, 15, 21 };
    std::vector<T> expectedOutputValues3 { 4, 10, 16, 22 };
    std::vector<T> expectedOutputValues4 { 5, 11, 17, 23 };
    std::vector<T> expectedOutputValues5 { 6, 12, 18, 24 };

    std::vector<std::vector<T>> expectedOutputValues{ expectedOutputValues0,
                                                      expectedOutputValues1,
                                                      expectedOutputValues2,
                                                      expectedOutputValues3,
                                                      expectedOutputValues4,
                                                      expectedOutputValues5 };

    UnpackTest<T>(tflite::BuiltinOperator_UNPACK,
                  tensorType,
                  backends,
                  inputShape,
                  expectedOutputShape,
                  inputValues,
                  expectedOutputValues,
                  2);
}

TEST_SUITE("Unpack_CpuRefTests")
{

// Fp32
TEST_CASE ("Unpack_Fp32_Axis0_Num4_CpuRef_Test")
{
std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
UnpackAxis0Num4Test<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("Unpack_Fp32_Axis2_Num6_CpuRef_Test")
{
std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
UnpackAxis2Num6Test<float>(tflite::TensorType_FLOAT32, backends);
}

// Uint8
TEST_CASE ("Unpack_Uint8_Axis0_Num4_CpuRef_Test")
{
std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
UnpackAxis0Num4Test<uint8_t>(tflite::TensorType_UINT8, backends);
}

TEST_CASE ("Unpack_Uint8_Axis2_Num6_CpuRef_Test")
{
std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
UnpackAxis2Num6Test<uint8_t>(tflite::TensorType_UINT8, backends);
}

} // End of Unpack_CpuRefTests

TEST_SUITE("Unpack_CpuAccTests")
{

// Fp32
TEST_CASE ("Unpack_Fp32_Axis0_Num4_CpuAcc_Test")
{
std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
UnpackAxis0Num4Test<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("Unpack_Fp32_Axis2_Num6_CpuAcc_Test")
{
std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
UnpackAxis2Num6Test<float>(tflite::TensorType_FLOAT32, backends);
}

// Uint8
TEST_CASE ("Unpack_Uint8_Axis0_Num4_CpuAcc_Test")
{
std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
UnpackAxis0Num4Test<uint8_t>(tflite::TensorType_UINT8, backends);
}

TEST_CASE ("Unpack_Uint8_Axis2_Num6_CpuAcc_Test")
{
std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
UnpackAxis2Num6Test<uint8_t>(tflite::TensorType_UINT8, backends);
}

} // End of Unpack_CpuAccTests

TEST_SUITE("Unpack_GpuAccTests")
{

// Fp32
TEST_CASE ("Unpack_Fp32_Axis0_Num4_GpuAcc_Test")
{
std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
UnpackAxis0Num4Test<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("Unpack_Fp32_Axis2_Num6_GpuAcc_Test")
{
std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
UnpackAxis2Num6Test<float>(tflite::TensorType_FLOAT32, backends);
}

// Uint8
TEST_CASE ("Unpack_Uint8_Axis0_Num4_GpuAcc_Test")
{
std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
UnpackAxis0Num4Test<uint8_t>(tflite::TensorType_UINT8, backends);
}

TEST_CASE ("Unpack_Uint8_Axis2_Num6_GpuAcc_Test")
{
std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
UnpackAxis2Num6Test<uint8_t>(tflite::TensorType_UINT8, backends);
}

} // End of Unpack_GpuAccTests

// End of Unpack Test Suite

} // namespace armnnDelegate