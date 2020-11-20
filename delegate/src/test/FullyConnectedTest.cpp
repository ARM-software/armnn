//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FullyConnectedTestHelper.hpp"

namespace
{

void FullyConnectedFp32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputTensorShape   { 1, 4, 1, 1 };
    std::vector<int32_t> weightsTensorShape { 1, 4 };
    std::vector<int32_t> biasTensorShape    { 1 };
    std::vector<int32_t> outputTensorShape  { 1, 1 };

    std::vector<float> inputValues = { 10, 20, 30, 40 };
    std::vector<float> weightsData = { 2, 3, 4, 5 };

    std::vector<float> expectedOutputValues = { (400 + 10) };

    // bias is set std::vector<float> biasData = { 10 } in the model
    FullyConnectedTest<float>(backends,
                              ::tflite::TensorType_FLOAT32,
                              tflite::ActivationFunctionType_NONE,
                              inputTensorShape,
                              weightsTensorShape,
                              biasTensorShape,
                              outputTensorShape,
                              inputValues,
                              expectedOutputValues,
                              weightsData);
}

void FullyConnectedActicationTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputTensorShape   { 1, 4, 1, 1 };
    std::vector<int32_t> weightsTensorShape { 1, 4 };
    std::vector<int32_t> biasTensorShape    { 1 };
    std::vector<int32_t> outputTensorShape  { 1, 1 };

    std::vector<float> inputValues = { -10, 20, 30, 40 };
    std::vector<float> weightsData = { 2, 3, 4, -5 };

    std::vector<float> expectedOutputValues = { 0 };

    // bias is set std::vector<float> biasData = { 10 } in the model
    FullyConnectedTest<float>(backends,
                              ::tflite::TensorType_FLOAT32,
                              tflite::ActivationFunctionType_RELU,
                              inputTensorShape,
                              weightsTensorShape,
                              biasTensorShape,
                              outputTensorShape,
                              inputValues,
                              expectedOutputValues,
                              weightsData);
}

void FullyConnectedInt8Test(std::vector<armnn::BackendId>& backends)
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
    FullyConnectedTest<int8_t>(backends,
                                ::tflite::TensorType_INT8,
                                tflite::ActivationFunctionType_NONE,
                                inputTensorShape,
                                weightsTensorShape,
                                biasTensorShape,
                                outputTensorShape,
                                inputValues,
                                expectedOutputValues,
                                weightsData);
}

TEST_SUITE("FullyConnected_GpuAccTests")
{

TEST_CASE ("FullyConnected_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    FullyConnectedFp32Test(backends);
}

TEST_CASE ("FullyConnected_Int8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    FullyConnectedInt8Test(backends);
}

TEST_CASE ("FullyConnected_Activation_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    FullyConnectedActicationTest(backends);
}

} // End of TEST_SUITE("FullyConnected_GpuAccTests")

TEST_SUITE("FullyConnected_CpuAccTests")
{

TEST_CASE ("FullyConnected_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    FullyConnectedFp32Test(backends);
}

TEST_CASE ("FullyConnected_Int8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    FullyConnectedInt8Test(backends);
}

TEST_CASE ("FullyConnected_Activation_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    FullyConnectedActicationTest(backends);
}

} // End of TEST_SUITE("FullyConnected_CpuAccTests")

TEST_SUITE("FullyConnected_CpuRefTests")
{

TEST_CASE ("FullyConnected_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    FullyConnectedFp32Test(backends);
}

TEST_CASE ("FullyConnected_Int8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    FullyConnectedInt8Test(backends);
}

TEST_CASE ("FullyConnected_Activation_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    FullyConnectedActicationTest(backends);
}

} // End of TEST_SUITE("FullyConnected_CpuRefTests")

} // anonymous namespace