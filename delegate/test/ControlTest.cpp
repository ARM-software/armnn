//
// Copyright © 2020,2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ControlTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

// CONCATENATION Operator
void ConcatUint8TwoInputsTest()
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
                               inputShape,
                               expectedOutputShape,
                               inputValues,
                               expectedOutputValues);
}

void ConcatInt16TwoInputsTest()
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
                               inputShape,
                               expectedOutputShape,
                               inputValues,
                               expectedOutputValues);
}

void ConcatFloat32TwoInputsTest()
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
                             inputShape,
                             expectedOutputShape,
                             inputValues,
                             expectedOutputValues);
}

void ConcatThreeInputsTest()
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
                               inputShape,
                               expectedOutputShape,
                               inputValues,
                               expectedOutputValues);
}

void ConcatAxisTest()
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
                               inputShape,
                               expectedOutputShape,
                               inputValues,
                               expectedOutputValues,
                               2);
}

// MEAN Operator
void MeanUint8KeepDimsTest()
{
    std::vector<int32_t> input0Shape { 1, 3 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 1, 1 };

    std::vector<uint8_t> input0Values { 5, 10, 15 }; // Inputs
    std::vector<int32_t> input1Values { 1 }; // Axis

    std::vector<uint8_t> expectedOutputValues { 10 };

    MeanTest<uint8_t>(tflite::BuiltinOperator_MEAN,
                      ::tflite::TensorType_UINT8,
                      input0Shape,
                      input1Shape,
                      expectedOutputShape,
                      input0Values,
                      input1Values,
                      expectedOutputValues,
                      true);
}

void MeanUint8Test()
{
    std::vector<int32_t> input0Shape { 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 2, 2 };

    std::vector<uint8_t> input0Values { 5, 10, 15, 20 }; // Inputs
    std::vector<int32_t> input1Values { 0 }; // Axis

    std::vector<uint8_t> expectedOutputValues { 5, 10, 15, 20 };

    MeanTest<uint8_t>(tflite::BuiltinOperator_MEAN,
                      ::tflite::TensorType_UINT8,
                      input0Shape,
                      input1Shape,
                      expectedOutputShape,
                      input0Values,
                      input1Values,
                      expectedOutputValues,
                      false);
}

void MeanFp32KeepDimsTest()
{
    std::vector<int32_t> input0Shape { 1, 2, 2 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 1, 1, 2 };

    std::vector<float>   input0Values { 1.0f, 1.5f, 2.0f, 2.5f }; // Inputs
    std::vector<int32_t> input1Values { 1 }; // Axis

    std::vector<float>   expectedOutputValues { 1.5f, 2.0f };

    MeanTest<float>(tflite::BuiltinOperator_MEAN,
                    ::tflite::TensorType_FLOAT32,
                    input0Shape,
                    input1Shape,
                    expectedOutputShape,
                    input0Values,
                    input1Values,
                    expectedOutputValues,
                    true);
}

void MeanFp32Test()
{
    std::vector<int32_t> input0Shape { 1, 2, 2, 1 };
    std::vector<int32_t> input1Shape { 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 1 };

    std::vector<float>   input0Values { 1.0f, 1.5f, 2.0f, 2.5f }; // Inputs
    std::vector<int32_t> input1Values { 2 }; // Axis

    std::vector<float>   expectedOutputValues { 1.25f, 2.25f };

    MeanTest<float>(tflite::BuiltinOperator_MEAN,
                    ::tflite::TensorType_FLOAT32,
                    input0Shape,
                    input1Shape,
                    expectedOutputShape,
                    input0Values,
                    input1Values,
                    expectedOutputValues,
                    false);
}

// CONCATENATION Tests.
TEST_SUITE("Concatenation_Tests")
{

TEST_CASE ("Concatenation_Uint8_Test")
{
    ConcatUint8TwoInputsTest();
}

TEST_CASE ("Concatenation_Int16_Test")
{
    ConcatInt16TwoInputsTest();
}

TEST_CASE ("Concatenation_Float32_Test")
{
    ConcatFloat32TwoInputsTest();
}

TEST_CASE ("Concatenation_Three_Inputs_Test")
{
    ConcatThreeInputsTest();
}

TEST_CASE ("Concatenation_Axis_Test")
{
    ConcatAxisTest();
}

}

// MEAN Tests
TEST_SUITE("Mean_Tests")
{

TEST_CASE ("Mean_Uint8_KeepDims_Test")
{
    MeanUint8KeepDimsTest();
}

TEST_CASE ("Mean_Uint8_Test")
{
    MeanUint8Test();
}

TEST_CASE ("Mean_Fp32_KeepDims_Test")
{
    MeanFp32KeepDimsTest();
}

TEST_CASE ("Mean_Fp32_Test")
{
    MeanFp32Test();
}

}

} // namespace armnnDelegate