//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ControlTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

// CONCATENATION Operator
void ConcatUint8TwoInputsTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 2, 2 };
    std::vector<int32_t> expectedOutputShape { 4, 2 };

    // Set input and output data
    std::vector<std::vector<uint8_t>> inputValues;
    std::vector<uint8_t> inputValue1 { 0, 1, 2, 3 }; // Lower bounds
    std::vector<uint8_t> inputValue2 { 252, 253, 254, 255 }; // Upper bounds
    inputValues.push_back(inputValue1);
    inputValues.push_back(inputValue2);

    std::vector<uint8_t> expectedOutputValues { 0, 1, 2, 3, 252, 253, 254, 255 };

    ConcatenationTest<uint8_t>(tflite::BuiltinOperator_CONCATENATION,
                               ::tflite::TensorType_UINT8,
                               backends,
                               inputShape,
                               expectedOutputShape,
                               inputValues,
                               expectedOutputValues);
}

void ConcatInt16TwoInputsTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 2, 2 };
    std::vector<int32_t> expectedOutputShape { 4, 2 };

    std::vector<std::vector<int16_t>> inputValues;
    std::vector<int16_t> inputValue1 { -32768, -16384, -1, 0 };
    std::vector<int16_t> inputValue2 { 1, 2, 16384, 32767 };
    inputValues.push_back(inputValue1);
    inputValues.push_back(inputValue2);

    std::vector<int16_t> expectedOutputValues { -32768, -16384, -1, 0, 1, 2, 16384, 32767};

    ConcatenationTest<int16_t>(tflite::BuiltinOperator_CONCATENATION,
                               ::tflite::TensorType_INT16,
                               backends,
                               inputShape,
                               expectedOutputShape,
                               inputValues,
                               expectedOutputValues);
}

void ConcatFloat32TwoInputsTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 2, 2 };
    std::vector<int32_t> expectedOutputShape { 4, 2 };

    std::vector<std::vector<float>> inputValues;
    std::vector<float> inputValue1 { -127.f, -126.f, -1.f, 0.f };
    std::vector<float> inputValue2 { 1.f, 2.f, 126.f, 127.f };
    inputValues.push_back(inputValue1);
    inputValues.push_back(inputValue2);

    std::vector<float> expectedOutputValues { -127.f, -126.f, -1.f, 0.f, 1.f, 2.f, 126.f, 127.f };

    ConcatenationTest<float>(tflite::BuiltinOperator_CONCATENATION,
                             ::tflite::TensorType_FLOAT32,
                             backends,
                             inputShape,
                             expectedOutputShape,
                             inputValues,
                             expectedOutputValues);
}

void ConcatThreeInputsTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 2, 2 };
    std::vector<int32_t> expectedOutputShape { 6, 2 };

    std::vector<std::vector<uint8_t>> inputValues;
    std::vector<uint8_t> inputValue1 { 0, 1, 2, 3 };
    std::vector<uint8_t> inputValue2 { 125, 126, 127, 128 };
    std::vector<uint8_t> inputValue3 { 252, 253, 254, 255 };
    inputValues.push_back(inputValue1);
    inputValues.push_back(inputValue2);
    inputValues.push_back(inputValue3);

    std::vector<uint8_t> expectedOutputValues { 0, 1, 2, 3, 125, 126, 127, 128, 252, 253, 254, 255 };

    ConcatenationTest<uint8_t>(tflite::BuiltinOperator_CONCATENATION,
                               ::tflite::TensorType_UINT8,
                               backends,
                               inputShape,
                               expectedOutputShape,
                               inputValues,
                               expectedOutputValues);
}

void ConcatAxisTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 1, 2, 2 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 4 };

    std::vector<std::vector<uint8_t>> inputValues;
    std::vector<uint8_t> inputValue1 { 0, 1, 2, 3 };
    std::vector<uint8_t> inputValue3 { 252, 253, 254, 255 };
    inputValues.push_back(inputValue1);
    inputValues.push_back(inputValue3);

    std::vector<uint8_t> expectedOutputValues { 0, 1, 252, 253, 2, 3, 254, 255 };

    ConcatenationTest<uint8_t>(tflite::BuiltinOperator_CONCATENATION,
                               ::tflite::TensorType_UINT8,
                               backends,
                               inputShape,
                               expectedOutputShape,
                               inputValues,
                               expectedOutputValues,
                               2);
}

// MEAN Operator
void MeanUint8KeepDimsTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 3 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 1, 1 };

    std::vector<uint8_t> input0Values { 5, 10, 15 }; // Inputs
    std::vector<int32_t> input1Values { 1 }; // Axis

    std::vector<uint8_t> expectedOutputValues { 10 };

    MeanTest<uint8_t>(tflite::BuiltinOperator_MEAN,
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

void MeanUint8Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 2, 2 };

    std::vector<uint8_t> input0Values { 5, 10, 15, 20 }; // Inputs
    std::vector<int32_t> input1Values { 0 }; // Axis

    std::vector<uint8_t> expectedOutputValues { 5, 10, 15, 20 };

    MeanTest<uint8_t>(tflite::BuiltinOperator_MEAN,
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

void MeanFp32KeepDimsTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 1, 1, 2 };

    std::vector<float>   input0Values { 1.0f, 1.5f, 2.0f, 2.5f }; // Inputs
    std::vector<int32_t> input1Values { 1 }; // Axis

    std::vector<float>   expectedOutputValues { 1.5f, 2.0f };

    MeanTest<float>(tflite::BuiltinOperator_MEAN,
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

void MeanFp32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 1 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 1 };

    std::vector<float>   input0Values { 1.0f, 1.5f, 2.0f, 2.5f }; // Inputs
    std::vector<int32_t> input1Values { 2 }; // Axis

    std::vector<float>   expectedOutputValues { 1.25f, 2.25f };

    MeanTest<float>(tflite::BuiltinOperator_MEAN,
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

// CONCATENATION Tests.
TEST_SUITE("Concatenation_CpuAccTests")
{

TEST_CASE ("Concatenation_Uint8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    ConcatUint8TwoInputsTest(backends);
}

TEST_CASE ("Concatenation_Int16_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    ConcatInt16TwoInputsTest(backends);
}

TEST_CASE ("Concatenation_Float32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    ConcatFloat32TwoInputsTest(backends);
}

TEST_CASE ("Concatenation_Three_Inputs_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    ConcatThreeInputsTest(backends);
}

TEST_CASE ("Concatenation_Axis_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    ConcatAxisTest(backends);
}

}

TEST_SUITE("Concatenation_GpuAccTests")
{

TEST_CASE ("Concatenation_Uint8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    ConcatUint8TwoInputsTest(backends);
}

TEST_CASE ("Concatenation_Int16_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    ConcatInt16TwoInputsTest(backends);
}

TEST_CASE ("Concatenation_Float32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    ConcatFloat32TwoInputsTest(backends);
}

TEST_CASE ("Concatenation_Three_Inputs_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    ConcatThreeInputsTest(backends);
}

TEST_CASE ("Concatenation_Axis_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    ConcatAxisTest(backends);
}

}

TEST_SUITE("Concatenation_CpuRefTests")
{

TEST_CASE ("Concatenation_Uint8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    ConcatUint8TwoInputsTest(backends);
}

TEST_CASE ("Concatenation_Int16_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    ConcatInt16TwoInputsTest(backends);
}

TEST_CASE ("Concatenation_Float32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    ConcatFloat32TwoInputsTest(backends);
}

TEST_CASE ("Concatenation_Three_Inputs_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    ConcatThreeInputsTest(backends);
}

TEST_CASE ("Concatenation_Axis_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    ConcatAxisTest(backends);
}

}

// MEAN Tests
TEST_SUITE("Mean_CpuAccTests")
{

TEST_CASE ("Mean_Uint8_KeepDims_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    MeanUint8KeepDimsTest(backends);
}

TEST_CASE ("Mean_Uint8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    MeanUint8Test(backends);
}

TEST_CASE ("Mean_Fp32_KeepDims_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    MeanFp32KeepDimsTest(backends);
}

TEST_CASE ("Mean_Fp32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    MeanFp32Test(backends);
}

}

TEST_SUITE("Mean_GpuAccTests")
{

TEST_CASE ("Mean_Uint8_KeepDims_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    MeanUint8KeepDimsTest(backends);
}

TEST_CASE ("Mean_Uint8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    MeanUint8Test(backends);
}

TEST_CASE ("Mean_Fp32_KeepDims_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    MeanFp32KeepDimsTest(backends);
}

TEST_CASE ("Mean_Fp32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    MeanFp32Test(backends);
}

}

TEST_SUITE("Mean_CpuRefTests")
{

TEST_CASE ("Mean_Uint8_KeepDims_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    MeanUint8KeepDimsTest(backends);
}

TEST_CASE ("Mean_Uint8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    MeanUint8Test(backends);
}

TEST_CASE ("Mean_Fp32_KeepDims_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    MeanFp32KeepDimsTest(backends);
}

TEST_CASE ("Mean_Fp32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    MeanFp32Test(backends);
}

}

} // namespace armnnDelegate