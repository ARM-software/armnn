//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PackTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

template <typename T>
void PackFp32Axis0Test(tflite::TensorType tensorType, std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 3, 2, 3 };
    std::vector<int32_t> expectedOutputShape { 2, 3, 2, 3 };

    std::vector<std::vector<T>> inputValues;
    inputValues.push_back(
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18
    });

    inputValues.push_back(
    {
        19, 20, 21,
        22, 23, 24,

        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36
    });

    std::vector<T> expectedOutputValues =
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18,


        19, 20, 21,
        22, 23, 24,

        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36
    };

    PackTest<T>(tflite::BuiltinOperator_PACK,
                tensorType,
                backends,
                inputShape,
                expectedOutputShape,
                inputValues,
                expectedOutputValues,
                0);
}

template <typename T>
void PackFp32Axis1Test(tflite::TensorType tensorType, std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 3, 2, 3 };
    std::vector<int32_t> expectedOutputShape { 3, 2, 2, 3 };

    std::vector<std::vector<T>> inputValues;
    inputValues.push_back(
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18
    });

    inputValues.push_back(
    {
        19, 20, 21,
        22, 23, 24,

        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36
    });

    std::vector<T> expectedOutputValues =
    {
        1, 2, 3,
        4, 5, 6,

        19, 20, 21,
        22, 23, 24,


        7, 8, 9,
        10, 11, 12,

        25, 26, 27,
        28, 29, 30,


        13, 14, 15,
        16, 17, 18,

        31, 32, 33,
        34, 35, 36
    };

    PackTest<T>(tflite::BuiltinOperator_PACK,
                tensorType,
                backends,
                inputShape,
                expectedOutputShape,
                inputValues,
                expectedOutputValues,
                1);
}

template <typename T>
void PackFp32Axis2Test(tflite::TensorType tensorType, std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 3, 2, 3 };
    std::vector<int32_t> expectedOutputShape { 3, 2, 2, 3 };

    std::vector<std::vector<T>> inputValues;
    inputValues.push_back(
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18
    });

    inputValues.push_back(
    {
        19, 20, 21,
        22, 23, 24,

        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36
    });

    std::vector<float> expectedOutputValues =
    {
        1, 2, 3,
        19, 20, 21,

        4, 5, 6,
        22, 23, 24,

        7, 8, 9,
        25, 26, 27,

        10, 11, 12,
        28, 29, 30,

        13, 14, 15,
        31, 32, 33,

        16, 17, 18,
        34, 35, 36
    };

    PackTest<T>(tflite::BuiltinOperator_PACK,
                tensorType,
                backends,
                inputShape,
                expectedOutputShape,
                inputValues,
                expectedOutputValues,
                2);
}

template <typename T>
void PackFp32Axis3Test(tflite::TensorType tensorType, std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 3, 2, 3 };
    std::vector<int32_t> expectedOutputShape { 3, 2, 3, 2 };

    std::vector<std::vector<T>> inputValues;
    inputValues.push_back(
    {
        1, 2, 3,
        4, 5, 6,

        7, 8, 9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18
    });

    inputValues.push_back(
    {
        19, 20, 21,
        22, 23, 24,

        25, 26, 27,
        28, 29, 30,

        31, 32, 33,
        34, 35, 36
    });

    std::vector<T> expectedOutputValues =
    {
        1, 19,
        2, 20,
        3, 21,

        4, 22,
        5, 23,
        6, 24,


        7, 25,
        8, 26,
        9, 27,

        10, 28,
        11, 29,
        12, 30,


        13, 31,
        14, 32,
        15, 33,

        16, 34,
        17, 35,
        18, 36
    };

    PackTest<T>(tflite::BuiltinOperator_PACK,
                tflite::TensorType_FLOAT32,
                backends,
                inputShape,
                expectedOutputShape,
                inputValues,
                expectedOutputValues,
                3);
}

template <typename T>
void PackFp32Inputs3Test(tflite::TensorType tensorType, std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 3, 3 };
    std::vector<int32_t> expectedOutputShape { 3, 3, 3 };

    std::vector<std::vector<T>> inputValues;
    inputValues.push_back(
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });

    inputValues.push_back(
    {
        10, 11, 12,
        13, 14, 15,
        16, 17, 18
    });

    inputValues.push_back(
    {
        19, 20, 21,
        22, 23, 24,
        25, 26, 27
    });

    std::vector<T> expectedOutputValues =
    {
        1, 2, 3,
        10, 11, 12,
        19, 20, 21,

        4, 5, 6,
        13, 14, 15,
        22, 23, 24,

        7, 8, 9,
        16, 17, 18,
        25, 26, 27
    };

    PackTest<T>(tflite::BuiltinOperator_PACK,
                tensorType,
                backends,
                inputShape,
                expectedOutputShape,
                inputValues,
                expectedOutputValues,
                1);
}

TEST_SUITE("Pack_CpuAccTests")
{

// Fp32
TEST_CASE ("Pack_Fp32_Axis0_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    PackFp32Axis0Test<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("Pack_Fp32_Axis1_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    PackFp32Axis1Test<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("Pack_Fp32_Axis2_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    PackFp32Axis2Test<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("Pack_Fp32_Axis3_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    PackFp32Axis3Test<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("Pack_Fp32_Inputs3_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    PackFp32Inputs3Test<float>(tflite::TensorType_FLOAT32, backends);
}

// Uint8
TEST_CASE ("Pack_Uint8_Axis0_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    PackFp32Axis0Test<uint8_t>(tflite::TensorType_UINT8, backends);
}

TEST_CASE ("Pack_Uint8_Inputs3_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    PackFp32Inputs3Test<uint8_t>(tflite::TensorType_UINT8, backends);
}

// Uint8
TEST_CASE ("Pack_Int8_Axis0_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    PackFp32Axis0Test<int8_t>(tflite::TensorType_INT8, backends);
}

TEST_CASE ("Pack_Int8_Inputs3_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    PackFp32Inputs3Test<int8_t>(tflite::TensorType_INT8, backends);
}

}

TEST_SUITE("Pack_GpuAccTests")
{

// Fp32
TEST_CASE ("Pack_Fp32_Axis0_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    PackFp32Axis0Test<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("Pack_Fp32_Axis1_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    PackFp32Axis1Test<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("Pack_Fp32_Axis2_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    PackFp32Axis2Test<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("Pack_Fp32_Axis3_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    PackFp32Axis3Test<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("Pack_Fp32_Inputs3_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    PackFp32Inputs3Test<float>(tflite::TensorType_FLOAT32, backends);
}

// Uint8
TEST_CASE ("Pack_Uint8_Axis0_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    PackFp32Axis0Test<uint8_t>(tflite::TensorType_UINT8, backends);
}

TEST_CASE ("Pack_Uint8_Inputs3_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    PackFp32Inputs3Test<uint8_t>(tflite::TensorType_UINT8, backends);
}

// Int8
TEST_CASE ("Pack_Int8_Axis0_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    PackFp32Axis0Test<int8_t>(tflite::TensorType_INT8, backends);
}

TEST_CASE ("Pack_Int8_Inputs3_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    PackFp32Inputs3Test<int8_t>(tflite::TensorType_INT8, backends);
}

}

TEST_SUITE("Pack_CpuRefTests")
{

// Fp32
TEST_CASE ("Pack_Fp32_Axis0_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    PackFp32Axis0Test<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("Pack_Fp32_Axis1_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    PackFp32Axis1Test<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("Pack_Fp32_Axis2_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    PackFp32Axis2Test<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("Pack_Fp32_Axis3_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    PackFp32Axis3Test<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("Pack_Fp32_Inputs3_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    PackFp32Inputs3Test<float>(tflite::TensorType_FLOAT32, backends);
}

// Uint8
TEST_CASE ("Pack_Uint8_Axis0_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    PackFp32Axis0Test<uint8_t>(tflite::TensorType_UINT8, backends);
}

TEST_CASE ("Pack_Uint8_Inputs3_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    PackFp32Inputs3Test<uint8_t>(tflite::TensorType_UINT8, backends);
}

// Int8
TEST_CASE ("Pack_Int8_Axis0_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    PackFp32Axis0Test<int8_t>(tflite::TensorType_INT8, backends);
}

TEST_CASE ("Pack_Int8_Inputs3_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    PackFp32Inputs3Test<int8_t>(tflite::TensorType_INT8, backends);
}

}

} // namespace armnnDelegate