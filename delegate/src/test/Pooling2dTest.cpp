//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Pooling2dTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void MaxPool2dFP32PaddingValidTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 1, 2, 1 };

    std::vector<float> inputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                       8.0f, 12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, -11.0f };

    std::vector<float> expectedOutputValues = { 12.0f, 7.0f };

    Pooling2dTest<float>(tflite::BuiltinOperator_MAX_POOL_2D,
                         ::tflite::TensorType_FLOAT32,
                         backends,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         ::tflite::Padding_VALID,
                         2,
                         2,
                         2,
                         2);
}

void MaxPool2dInt8PaddingValidTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 1, 2, 1 };

    std::vector<int8_t > inputValues = { -5, 8, -10, 7,
                                         8, 12, -15, 2,
                                         3, -4, -1, -11 };

    std::vector<int8_t> expectedOutputValues = { 12, 7 };

    Pooling2dTest<int8_t>(tflite::BuiltinOperator_MAX_POOL_2D,
                          ::tflite::TensorType_INT8,
                          backends,
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          ::tflite::Padding_VALID,
                          2,
                          2,
                          2,
                          2,
                          tflite::ActivationFunctionType_NONE,
                          2.5f,
                          1);
}

void MaxPool2dFP32PaddingSameTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    std::vector<float> inputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                       8.0f, 12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, -11.0f };

    std::vector<float> expectedOutputValues = { 12.0f, 7.0f, 3.0f, -1.0f };

    Pooling2dTest<float>(tflite::BuiltinOperator_MAX_POOL_2D,
                         ::tflite::TensorType_FLOAT32,
                         backends,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         ::tflite::Padding_SAME,
                         2,
                         2,
                         2,
                         2);
}

void MaxPool2dInt8PaddingSameTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    std::vector<int8_t> inputValues = { -5, 8, -10, 7,
                                        8, 12, -15, 2,
                                        3, -4, -1, -11 };

    std::vector<int8_t> expectedOutputValues = { 12, 7, 3, -1 };

    Pooling2dTest<int8_t>(tflite::BuiltinOperator_MAX_POOL_2D,
                          ::tflite::TensorType_INT8,
                          backends,
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          ::tflite::Padding_SAME,
                          2,
                          2,
                          2,
                          2,
                          tflite::ActivationFunctionType_NONE,
                          2.5f,
                          1);
}

void MaxPool2dFP32ReluTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 3, 1 };

    std::vector<float> inputValues = { -5.0f, -8.0f, -10.0f, 7.0f,
                                       -8.0f, -12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, -11.0f };

    std::vector<float> expectedOutputValues = { 0.0f, 0.0f, 7.0f, 3.0f, 0.0f, 2.0f };

    Pooling2dTest<float>(tflite::BuiltinOperator_MAX_POOL_2D,
                         ::tflite::TensorType_FLOAT32,
                         backends,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         ::tflite::Padding_VALID,
                         1,
                         1,
                         2,
                         2,
                         ::tflite::ActivationFunctionType_RELU);
}

void MaxPool2dInt8ReluTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 3, 1 };

    std::vector<int8_t> inputValues = { -5, -8, -10, 7,
                                        -8, -12, -15, 2,
                                        3, -4, -1, -11 };

    std::vector<int8_t> expectedOutputValues = { 1, 1, 7, 3, 1, 2 };

    Pooling2dTest<int8_t>(tflite::BuiltinOperator_MAX_POOL_2D,
                          ::tflite::TensorType_INT8,
                          backends,
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          ::tflite::Padding_VALID,
                          1,
                          1,
                          2,
                          2,
                          ::tflite::ActivationFunctionType_RELU,
                          2.0f,
                          1);
}

void MaxPool2dFP32Relu6Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    std::vector<float> inputValues = { -5.0f, -8.0f, -10.0f, 7.0f,
                                       -8.0f, -12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, -11.0f };

    std::vector<float> expectedOutputValues = { 0.0f, 0.0f, 3.0f, 0.0f };

    Pooling2dTest<float>(tflite::BuiltinOperator_MAX_POOL_2D,
                         ::tflite::TensorType_FLOAT32,
                         backends,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         ::tflite::Padding_SAME,
                         2,
                         2,
                         1,
                         1,
                         ::tflite::ActivationFunctionType_RELU6);
}

void MaxPool2dInt8Relu6Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    std::vector<int8_t> inputValues = { -5, -8, -10, 7,
                                        -8, -12, -15, 2,
                                        3, -4, -1, -11 };

    std::vector<int8_t> expectedOutputValues = { 1, 1, 3, 1 };

    Pooling2dTest<int8_t>(tflite::BuiltinOperator_MAX_POOL_2D,
                          ::tflite::TensorType_INT8,
                          backends,
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          ::tflite::Padding_SAME,
                          2,
                          2,
                          1,
                          1,
                          ::tflite::ActivationFunctionType_RELU6,
                          2.0f,
                          1);
}

TEST_SUITE("Pooling2dTest")
{

TEST_CASE ("MaxPooling2d_FP32_PaddingValid_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dFP32PaddingValidTest(backends);
}

TEST_CASE ("MaxPooling2d_FP32_PaddingValid_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dFP32PaddingValidTest(backends);
}

TEST_CASE ("MaxPooling2d_Int8_PaddingValid_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dInt8PaddingValidTest(backends);
}

TEST_CASE ("MaxPooling2d_Int8_PaddingValid_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dInt8PaddingValidTest(backends);
}

TEST_CASE ("MaxPooling2d_FP32_PaddingSame_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dFP32PaddingSameTest(backends);
}

TEST_CASE ("MaxPooling2d_FP32_PaddingSame_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dFP32PaddingSameTest(backends);
}

TEST_CASE ("MaxPooling2d_Int8_PaddingSame_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dInt8PaddingSameTest(backends);
}

TEST_CASE ("MaxPooling2d_Int8_PaddingSame_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dInt8PaddingSameTest(backends);
}

TEST_CASE ("MaxPooling2d_FP32_Relu_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dFP32ReluTest(backends);
}

TEST_CASE ("MaxPooling2d_FP32_Relu_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dFP32ReluTest(backends);
}

TEST_CASE ("MaxPooling2d_Int8_Relu_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dInt8ReluTest(backends);
}

TEST_CASE ("MaxPooling2d_Int8_Relu_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dInt8ReluTest(backends);
}

TEST_CASE ("MaxPooling2d_FP32_Relu6_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dFP32Relu6Test(backends);
}

TEST_CASE ("MaxPooling2d_FP32_Relu6_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dFP32Relu6Test(backends);
}

TEST_CASE ("MaxPooling2d_Int8_Relu6_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dInt8Relu6Test(backends);
}

TEST_CASE ("MaxPooling2d_Int8_Relu6_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc,
                                               armnn::Compute::CpuRef };
    MaxPool2dInt8Relu6Test(backends);
}

}

} // namespace armnnDelegate