//
// Copyright © 2020-2021,2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FullyConnectedTestHelper.hpp"

#include <doctest/doctest.h>

namespace
{

void FullyConnectedFp32Test(const std::vector<armnn::BackendId>& backends = {}, bool constantWeights = true)
{
    std::vector<int32_t> inputTensorShape   { 1, 4, 1, 1 };
    std::vector<int32_t> weightsTensorShape { 1, 4 };
    std::vector<int32_t> biasTensorShape    { 1 };
    std::vector<int32_t> outputTensorShape  { 1, 1 };

    std::vector<float> inputValues = { 10, 20, 30, 40 };
    std::vector<float> weightsData = { 2, 3, 4, 5 };

    std::vector<float> expectedOutputValues = { (400 + 10) };

    // bias is set std::vector<float> biasData = { 10 } in the model
    FullyConnectedTest<float>(::tflite::TensorType_FLOAT32,
                              tflite::ActivationFunctionType_NONE,
                              inputTensorShape,
                              weightsTensorShape,
                              biasTensorShape,
                              outputTensorShape,
                              inputValues,
                              expectedOutputValues,
                              weightsData,
                              backends,
                              constantWeights);
}

void FullyConnectedActivationTest(const std::vector<armnn::BackendId>& backends = {}, bool constantWeights = true)
{
    std::vector<int32_t> inputTensorShape   { 1, 4, 1, 1 };
    std::vector<int32_t> weightsTensorShape { 1, 4 };
    std::vector<int32_t> biasTensorShape    { 1 };
    std::vector<int32_t> outputTensorShape  { 1, 1 };

    std::vector<float> inputValues = { -10, 20, 30, 40 };
    std::vector<float> weightsData = { 2, 3, 4, -5 };

    std::vector<float> expectedOutputValues = { 0 };

    // bias is set std::vector<float> biasData = { 10 } in the model
    FullyConnectedTest<float>(::tflite::TensorType_FLOAT32,
                              tflite::ActivationFunctionType_RELU,
                              inputTensorShape,
                              weightsTensorShape,
                              biasTensorShape,
                              outputTensorShape,
                              inputValues,
                              expectedOutputValues,
                              weightsData,
                              backends,
                              constantWeights);
}

void FullyConnectedInt8Test(const std::vector<armnn::BackendId>& backends = {}, bool constantWeights = true)
{
    std::vector<int32_t> inputTensorShape   { 1, 4, 2, 1 };
    std::vector<int32_t> weightsTensorShape { 1, 4 };
    std::vector<int32_t> biasTensorShape    { 1 };
    std::vector<int32_t> outputTensorShape  { 2, 1 };

    std::vector<int8_t> inputValues = { 1, 2, 3, 4, 5, 10, 15, 20 };
    std::vector<int8_t> weightsData = { 2, 3, 4, 5 };

    std::vector<int8_t> expectedOutputValues = { 25, 105 };  // (40 + 10) / 2, (200 + 10) / 2

    // bias is set std::vector<int32_t> biasData = { 10 } in the model
    // input and weights quantization scale 1.0f and offset 0 in the model
    // output quantization scale 2.0f and offset 0 in the model
    FullyConnectedTest<int8_t>(::tflite::TensorType_INT8,
                               tflite::ActivationFunctionType_NONE,
                                inputTensorShape,
                                weightsTensorShape,
                                biasTensorShape,
                               outputTensorShape,
                               inputValues,
                               expectedOutputValues,
                               weightsData,
                                backends,
                                constantWeights);
}

TEST_SUITE("FullyConnectedTests")
{

TEST_CASE ("FullyConnected_FP32_Test")
{
    FullyConnectedFp32Test();
}

TEST_CASE ("FullyConnected_Int8_Test")
{
    FullyConnectedInt8Test();
}

TEST_CASE ("FullyConnected_Activation_Test")
{
    FullyConnectedActivationTest();
}

TEST_CASE ("FullyConnected_Weights_As_Inputs_FP32_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    FullyConnectedFp32Test(backends, false);
}

TEST_CASE ("FullyConnected_Weights_As_Inputs_Int8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    FullyConnectedInt8Test(backends, false);
}

TEST_CASE ("FullyConnected_Weights_As_Inputs_Activation_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    FullyConnectedActivationTest(backends, false);
}

} // End of TEST_SUITE("FullyConnectedTests")

} // anonymous namespace