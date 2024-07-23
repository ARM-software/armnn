//
// Copyright © 2020, 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "QuantizationTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

// Dequantize operator test functions.
void DequantizeUint8Test(const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> inputShape  { 2, 4 };
    std::vector<int32_t> outputShape { 2, 4 };

    // Set input and output data
    std::vector<uint8_t> inputValues
    {
        0, 1, 2, 3, // Lower bounds
        252, 253, 254, 255 // Upper bounds
    };
    std::vector<float> expectedOutputValues
    {
        0.f, 1.f, 2.f, 3.f,
        252.f, 253.f, 254.f, 255.f
    };

    QuantizationTest<uint8_t, float>(tflite::BuiltinOperator_DEQUANTIZE,
                                     ::tflite::TensorType_UINT8,
                                     ::tflite::TensorType_FLOAT32,
                                     inputShape,
                                     outputShape,
                                     inputValues,
                                     expectedOutputValues,
                                     backends);
}

void DequantizeInt8Test(const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> inputShape  { 2, 4 };
    std::vector<int32_t> outputShape { 2, 4 };

    std::vector<int8_t> inputValues
    {
        -1, 0, 1, 2,
        -128, -127, 126, 127
    };
    std::vector<float> expectedOutputValues
    {
        -1.f, 0.f, 1.f, 2.f,
        -128.f, -127.f, 126.f, 127.f
    };

    QuantizationTest<int8_t , float>(tflite::BuiltinOperator_DEQUANTIZE,
                                     ::tflite::TensorType_INT8,
                                     ::tflite::TensorType_FLOAT32,
                                     inputShape,
                                     outputShape,
                                     inputValues,
                                     expectedOutputValues,
                                     backends);
}

void DequantizeInt16Test(const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> inputShape  { 2, 5 };
    std::vector<int32_t> outputShape { 2, 5 };

    std::vector<int16_t> inputValues
    {
        -1, 0, 1, 2,
        -32768, -16384, 16384, 32767
    };
    std::vector<float> expectedOutputValues
    {
        -1.f, 0.f, 1.f, 2.f,
        -32768.f, -16384.f, 16384.f, 32767.f
    };

    QuantizationTest<int16_t, float>(tflite::BuiltinOperator_DEQUANTIZE,
                                     ::tflite::TensorType_INT16,
                                     ::tflite::TensorType_FLOAT32,
                                     inputShape,
                                     outputShape,
                                     inputValues,
                                     expectedOutputValues,
                                     backends);
}

// Quantize operator test functions.
void QuantizeFloat32Uint8Test(const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> inputShape  { 2, 4 };
    std::vector<int32_t> outputShape { 2, 4 };

    // Set input and output data
    std::vector<float> inputValues
    {
         -1.f, 0.f, 1.f, 2.f, // Lower bounds
         252.f, 253.f, 255.f, 256.f // Upper bounds
    };
    std::vector<uint8_t> expectedOutputValues
    {
        0, 0, 1, 2,
        252, 253, 255, 255
    };

    QuantizationTest<float, uint8_t>(tflite::BuiltinOperator_QUANTIZE,
                                     ::tflite::TensorType_FLOAT32,
                                     ::tflite::TensorType_UINT8,
                                     inputShape,
                                     outputShape,
                                     inputValues,
                                     expectedOutputValues,
                                     backends);
}

void QuantizeFloat32Int8Test(const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> inputShape  { 2, 4 };
    std::vector<int32_t> outputShape { 2, 4 };

    std::vector<float> inputValues
    {
        -1.f, 0.f, 1.f, 2.f,
        -128.5f, -127.f, 126.f, 127.5f
    };
    std::vector<int8_t> expectedOutputValues
    {
        -1, 0, 1, 2,
        -128, -127, 126, 127
    };

    QuantizationTest<float, int8_t>(tflite::BuiltinOperator_QUANTIZE,
                                    ::tflite::TensorType_FLOAT32,
                                    ::tflite::TensorType_INT8,
                                    inputShape,
                                    outputShape,
                                    inputValues,
                                    expectedOutputValues,
                                    backends);
}

void QuantizeFloat32Int16Test(const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> inputShape  { 2, 4 };
    std::vector<int32_t> outputShape { 2, 4 };

    std::vector<float> inputValues
    {
        -1.f, 0.f, 1.f, 2.f,
        -32768.5f, -16384.f, 16384.f, 32767.5f
    };
    std::vector<int16_t> expectedOutputValues
    {
        -1, 0, 1, 2,
        -32768, -16384, 16384, 32767
    };

    QuantizationTest<float, int16_t>(tflite::BuiltinOperator_QUANTIZE,
                                    ::tflite::TensorType_FLOAT32,
                                    ::tflite::TensorType_INT16,
                                    inputShape,
                                    outputShape,
                                    inputValues,
                                    expectedOutputValues,
                                    backends);
}

void QuantizeInt16Int16Test(const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> inputShape  { 2, 4 };
    std::vector<int32_t> outputShape { 2, 4 };

    std::vector<int16_t> inputValues
    {
        -1, 0, 1, 2,
        -32768, -16384, 16384, 32767
    };
    std::vector<int16_t> expectedOutputValues
    {
        -1, 0, 1, 2,
        -32768, -16384, 16384, 32767
    };

    QuantizationTest<int16_t, int16_t>(tflite::BuiltinOperator_QUANTIZE,
                                     ::tflite::TensorType_INT16,
                                     ::tflite::TensorType_INT16,
                                     inputShape,
                                     outputShape,
                                     inputValues,
                                     expectedOutputValues,
                                     backends);
}

void QuantizeInt16Int8Test(const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> inputShape  { 2, 4 };
    std::vector<int32_t> outputShape { 2, 4 };

    std::vector<int16_t> inputValues
    {
        -1, 0, 1, 2,
        -32768, -16384, 16384, 32767
    };
    std::vector<int8_t> expectedOutputValues
    {
        -1, 0, 1, 2,
        -128, -128, 127, 127
    };

    QuantizationTest<int16_t, int8_t>(tflite::BuiltinOperator_QUANTIZE,
                                      ::tflite::TensorType_INT16,
                                      ::tflite::TensorType_INT8,
                                      inputShape,
                                      outputShape,
                                      inputValues,
                                      expectedOutputValues,
                                      backends);
}

void QuantizeInt8Uint8Test(const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> inputShape  { 2, 4 };
    std::vector<int32_t> outputShape { 2, 4 };

    std::vector<int8_t> inputValues
    {
        -1, 0, 1, 2,
        -128, -127, 126, 127
    };
    std::vector<uint8_t> expectedOutputValues
    {
        0, 0, 1, 2,
        0, 0, 126, 127
    };

    QuantizationTest<int8_t, uint8_t>(tflite::BuiltinOperator_QUANTIZE,
                                      ::tflite::TensorType_INT8,
                                      ::tflite::TensorType_UINT8,
                                      inputShape,
                                      outputShape,
                                      inputValues,
                                      expectedOutputValues,
                                      backends);
}

void QuantizeUint8Int8Test(const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> inputShape  { 2, 4 };
    std::vector<int32_t> outputShape { 2, 4 };

    std::vector<uint8_t> inputValues
    {
        0, 1, 2, 3,
        126, 127, 254, 255
    };
    std::vector<int8_t> expectedOutputValues
    {
        0, 1, 2, 3,
        126, 127, 127, 127
    };

    QuantizationTest<uint8_t, int8_t>(tflite::BuiltinOperator_QUANTIZE,
                                      ::tflite::TensorType_UINT8,
                                      ::tflite::TensorType_INT8,
                                      inputShape,
                                      outputShape,
                                      inputValues,
                                      expectedOutputValues,
                                      backends);
}

TEST_SUITE("CpuRef_QuantizationTests")
{

TEST_CASE ("DEQUANTIZE_UINT8_Test")
{
    DequantizeUint8Test();
}


TEST_CASE ("DEQUANTIZE_INT8_Test")
{
    DequantizeInt8Test();
}


TEST_CASE ("DEQUANTIZE_INT16_Test")
{
    DequantizeInt16Test();
}


TEST_CASE ("QUANTIZE_FLOAT32_UINT8_Test")
{
    QuantizeFloat32Uint8Test();
}


TEST_CASE ("QUANTIZE_FLOAT32_INT8_Test")
{
    QuantizeFloat32Int8Test();
}


TEST_CASE ("QUANTIZE_FLOAT32_INT16_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    QuantizeFloat32Int16Test(backends);
}


TEST_CASE ("QUANTIZE_INT16_INT16_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    QuantizeInt16Int16Test(backends);
}


TEST_CASE ("QUANTIZE_INT16_INT8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    QuantizeInt16Int8Test(backends);
}



TEST_CASE ("QUANTIZE_INT8_UINT8_Test")
{
    QuantizeInt8Uint8Test();
}


TEST_CASE ("QUANTIZE_UINT8_INT8_Test")
{
    QuantizeUint8Int8Test();
}

}

} // namespace armnnDelegate