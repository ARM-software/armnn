//
// Copyright © 2020, 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Pooling2dTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

void MaxPool2dFP32PaddingValidTest(const std::vector<armnn::BackendId>& backends = {})
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
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         ::tflite::Padding_VALID,
                         2,
                         2,
                         2,
                         2);
}

void MaxPool2dInt8PaddingValidTest(const std::vector<armnn::BackendId>& backends = {})
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
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          backends,
                          ::tflite::Padding_VALID,
                          2,
                          2,
                          2,
                          2,
                          tflite::ActivationFunctionType_NONE,
                          2.5f,
                          1);
}

void MaxPool2dFP32PaddingSameTest(const std::vector<armnn::BackendId>& backends = {})
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
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         ::tflite::Padding_SAME,
                         2,
                         2,
                         2,
                         2);
}

void MaxPool2dInt8PaddingSameTest(const std::vector<armnn::BackendId>& backends = {})
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
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          backends,
                          ::tflite::Padding_SAME,
                          2,
                          2,
                          2,
                          2,
                          tflite::ActivationFunctionType_NONE,
                          2.5f,
                          1);
}

void MaxPool2dFP32ReluTest(const std::vector<armnn::BackendId>& backends = {})
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
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         ::tflite::Padding_VALID,
                         1,
                         1,
                         2,
                         2,
                         ::tflite::ActivationFunctionType_RELU);
}

void MaxPool2dInt8ReluTest(const std::vector<armnn::BackendId>& backends = {})
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
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          backends,
                          ::tflite::Padding_VALID,
                          1,
                          1,
                          2,
                          2,
                          ::tflite::ActivationFunctionType_RELU,
                          2.0f,
                          1);
}

void MaxPool2dFP32Relu6Test(const std::vector<armnn::BackendId>& backends = {})
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
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         ::tflite::Padding_SAME,
                         2,
                         2,
                         1,
                         1,
                         ::tflite::ActivationFunctionType_RELU6);
}

void MaxPool2dInt8Relu6Test(const std::vector<armnn::BackendId>& backends = {})
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
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          backends,
                          ::tflite::Padding_SAME,
                          2,
                          2,
                          1,
                          1,
                          ::tflite::ActivationFunctionType_RELU6,
                          2.0f,
                          1);
}

void MaxPool2dUint8PaddingSameTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    std::vector<uint8_t> inputValues = { 5, 8, 10, 7,
                                         8, 12, 15, 2,
                                         3, 4, 1, 11 };

    std::vector<uint8_t> expectedOutputValues = { 12, 15, 4, 11 };

    Pooling2dTest<uint8_t>(tflite::BuiltinOperator_MAX_POOL_2D,
                           ::tflite::TensorType_UINT8,
                           inputShape,
                           outputShape,
                           inputValues,
                           expectedOutputValues,
                           backends,
                           ::tflite::Padding_SAME,
                           2,
                           2,
                           2,
                           2,
                           tflite::ActivationFunctionType_NONE,
                           2.5f,
                           1);
}

void MaxPool2dUint8ReluTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 3, 1 };

    std::vector<uint8_t> inputValues = { 12, 8, 10, 15,
                                         8, 5, 7, 2,
                                         3, 4, 1, 11 };

    std::vector<uint8_t> expectedOutputValues = { 12, 10, 15, 8, 7, 11 };

    Pooling2dTest<uint8_t>(tflite::BuiltinOperator_MAX_POOL_2D,
                           ::tflite::TensorType_UINT8,
                           inputShape,
                           outputShape,
                           inputValues,
                           expectedOutputValues,
                           backends,
                           ::tflite::Padding_VALID,
                           1,
                           1,
                           2,
                           2,
                           ::tflite::ActivationFunctionType_RELU,
                           2.0f,
                           1);
}

void MaxPool2dInt16PaddingSameTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    std::vector<int16_t> inputValues = { -5, 8, -10, 7,
                                         8, 12, -15, 2,
                                         3, -4, -1, -11 };

    std::vector<int16_t> expectedOutputValues = { 12, 7, 3, -1 };

    Pooling2dTest<int16_t>(tflite::BuiltinOperator_MAX_POOL_2D,
                           ::tflite::TensorType_INT16,
                           inputShape,
                           outputShape,
                           inputValues,
                           expectedOutputValues,
                           backends,
                           ::tflite::Padding_SAME,
                           2,
                           2,
                           2,
                           2,
                           tflite::ActivationFunctionType_NONE,
                           2.5f,
                           0);
}

void MaxPool2dInt16ReluTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 3, 1 };

    std::vector<int16_t> inputValues = { -5, -8, -10, 7,
                                         -8, -12, -15, 2,
                                         3, -4, -1, -11 };

    std::vector<int16_t> expectedOutputValues = { 0, 0, 7, 3, 0, 2 };

    Pooling2dTest<int16_t>(tflite::BuiltinOperator_MAX_POOL_2D,
                           ::tflite::TensorType_INT16,
                           inputShape,
                           outputShape,
                           inputValues,
                           expectedOutputValues,
                           backends,
                           ::tflite::Padding_VALID,
                           1,
                           1,
                           2,
                           2,
                           ::tflite::ActivationFunctionType_RELU,
                           2.0f,
                           0);
}

void AveragePool2dFP32PaddingValidTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 1, 2, 1 };

    std::vector<float> inputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                       8.0f, 12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, -11.0f };

    std::vector<float> expectedOutputValues = { 5.75f, -4.0f };

    Pooling2dTest<float>(tflite::BuiltinOperator_AVERAGE_POOL_2D,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         ::tflite::Padding_VALID,
                         2,
                         2,
                         2,
                         2);
}

void AveragePool2dInt8PaddingValidTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 1, 2, 1 };

    std::vector<int8_t > inputValues = { -5, 8, -10, 7,
                                         8, 12, -15, 2,
                                         3, -4, -1, -11 };

    std::vector<int8_t> expectedOutputValues = { 6, -4 };

    Pooling2dTest<int8_t>(tflite::BuiltinOperator_AVERAGE_POOL_2D,
                          ::tflite::TensorType_INT8,
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          backends,
                          ::tflite::Padding_VALID,
                          2,
                          2,
                          2,
                          2,
                          tflite::ActivationFunctionType_NONE,
                          2.5f,
                          1);
}

void AveragePool2dFP32PaddingSameTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    std::vector<float> inputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                       8.0f, 12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, -11.0f };

    std::vector<float> expectedOutputValues = { 5.75f, -4.0f, -0.5f, -6.0f };

    Pooling2dTest<float>(tflite::BuiltinOperator_AVERAGE_POOL_2D,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         ::tflite::Padding_SAME,
                         2,
                         2,
                         2,
                         2);
}

void AveragePool2dInt8PaddingSameTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    std::vector<int8_t > inputValues = { -5, 8, -10, 7,
                                         8, 12, -15, 2,
                                         3, -4, -1, -11 };

    std::vector<int8_t> expectedOutputValues = { 6, -4, -1, -6 };

    Pooling2dTest<int8_t>(tflite::BuiltinOperator_AVERAGE_POOL_2D,
                          ::tflite::TensorType_INT8,
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          backends,
                          ::tflite::Padding_SAME,
                          2,
                          2,
                          2,
                          2,
                          tflite::ActivationFunctionType_NONE,
                          2.5f,
                          1);
}

void AveragePool2dFP32ReluTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 3, 1 };

    std::vector<float> inputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                       -8.0f, 12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, 11.0f };

    std::vector<float> expectedOutputValues = { 1.75f, 0.0f, 0.0f, 0.75f, 0.0f, 0.0f };

    Pooling2dTest<float>(tflite::BuiltinOperator_AVERAGE_POOL_2D,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         ::tflite::Padding_VALID,
                         1,
                         1,
                         2,
                         2,
                         ::tflite::ActivationFunctionType_RELU);
}

void AveragePool2dInt8ReluTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 3, 1 };

    std::vector<int8_t> inputValues = { -5, 8, -10, 7,
                                        -8, 12, -15, 2,
                                        3, -4, -1, 11 };

    std::vector<int8_t> expectedOutputValues = { 2, 1, 1, 1, 1, 1 };

    Pooling2dTest<int8_t>(tflite::BuiltinOperator_AVERAGE_POOL_2D,
                          ::tflite::TensorType_INT8,
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          backends,
                          ::tflite::Padding_VALID,
                          1,
                          1,
                          2,
                          2,
                          ::tflite::ActivationFunctionType_RELU,
                          2.5f,
                          1);
}

void AveragePool2dFP32Relu6Test(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    std::vector<float> inputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                       -8.0f, 12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, 11.0f };

    std::vector<float> expectedOutputValues = { 0.0f, 0.0f, 3.0f, 0.0f };

    Pooling2dTest<float>(tflite::BuiltinOperator_AVERAGE_POOL_2D,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         ::tflite::Padding_SAME,
                         2,
                         2,
                         1,
                         1,
                         ::tflite::ActivationFunctionType_RELU6);
}

void AveragePool2dInt8Relu6Test(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    std::vector<int8_t> inputValues = { -5, 8, -10, 7,
                                        -8, 12, -15, 2,
                                        3, -4, -1, 11 };

    std::vector<int8_t> expectedOutputValues = { 1, 1, 3, 1 };

    Pooling2dTest<int8_t>(tflite::BuiltinOperator_AVERAGE_POOL_2D,
                          ::tflite::TensorType_INT8,
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          backends,
                          ::tflite::Padding_SAME,
                          2,
                          2,
                          1,
                          1,
                          ::tflite::ActivationFunctionType_RELU6,
                          2.5f,
                          1);
}

void AveragePool2dUint8PaddingSameTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    std::vector<uint8_t> inputValues = { 5, 8, 10, 7,
                                         8, 12, 15, 2,
                                         3, 4, 1, 11 };

    std::vector<uint8_t> expectedOutputValues = { 8, 9, 4, 6 };

    Pooling2dTest<uint8_t>(tflite::BuiltinOperator_AVERAGE_POOL_2D,
                           ::tflite::TensorType_UINT8,
                           inputShape,
                           outputShape,
                           inputValues,
                           expectedOutputValues,
                           backends,
                           ::tflite::Padding_SAME,
                           2,
                           2,
                           2,
                           2,
                           tflite::ActivationFunctionType_NONE,
                           2.5f,
                           1);
}

void AveragePool2dUint8ReluTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 3, 1 };

    std::vector<uint8_t> inputValues = { 12, 8, 10, 15,
                                         8, 5, 7, 2,
                                         3, 4, 1, 11 };

    std::vector<uint8_t> expectedOutputValues = { 8, 8, 9, 5, 4, 5 };

    Pooling2dTest<uint8_t>(tflite::BuiltinOperator_AVERAGE_POOL_2D,
                           ::tflite::TensorType_UINT8,
                           inputShape,
                           outputShape,
                           inputValues,
                           expectedOutputValues,
                           backends,
                           ::tflite::Padding_VALID,
                           1,
                           1,
                           2,
                           2,
                           ::tflite::ActivationFunctionType_RELU,
                           2.0f,
                           1);
}

void AveragePool2dInt16PaddingSameTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    std::vector<int16_t > inputValues = { -5, 8, -10, 7,
                                         8, 12, -15, 2,
                                         3, -4, -1, -11 };

    std::vector<int16_t> expectedOutputValues = { 6, -4, -1, -6 };

    Pooling2dTest<int16_t>(tflite::BuiltinOperator_AVERAGE_POOL_2D,
                           ::tflite::TensorType_INT16,
                           inputShape,
                           outputShape,
                           inputValues,
                           expectedOutputValues,
                           backends,
                           ::tflite::Padding_SAME,
                           2,
                           2,
                           2,
                           2,
                           tflite::ActivationFunctionType_NONE,
                           2.5f,
                           0);
}

void AveragePool2dInt16ReluTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 3, 1 };

    std::vector<int16_t> inputValues = { -5, 8, -10, 7,
                                         -8, 12, -15, 2,
                                         3, -4, -1, 11 };

    std::vector<int16_t> expectedOutputValues = { 2, 0, 0, 1, 0, 0 };

    Pooling2dTest<int16_t>(tflite::BuiltinOperator_AVERAGE_POOL_2D,
                           ::tflite::TensorType_INT16,
                           inputShape,
                           outputShape,
                           inputValues,
                           expectedOutputValues,
                           backends,
                           ::tflite::Padding_VALID,
                           1,
                           1,
                           2,
                           2,
                           ::tflite::ActivationFunctionType_RELU,
                           2.5f,
                           0);
}

void L2Pool2dFP32PaddingValidTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 1, 2, 1 };

    std::vector<float> inputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                       8.0f, 12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, -11.0f };

    std::vector<float> expectedOutputValues = { 8.616844f, 9.721111f };

    Pooling2dTest<float>(tflite::BuiltinOperator_L2_POOL_2D,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         ::tflite::Padding_VALID,
                         2,
                         2,
                         2,
                         2);
}

void L2Pool2dFP32PaddingSameTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    std::vector<float> inputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                       8.0f, 12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, -11.0f };

    std::vector<float> expectedOutputValues = { 8.616844f, 9.721111f, 3.535534f, 7.81025f };

    Pooling2dTest<float>(tflite::BuiltinOperator_L2_POOL_2D,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         ::tflite::Padding_SAME,
                         2,
                         2,
                         2,
                         2);
}

void L2Pool2dFP32ReluTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 3, 1 };

    std::vector<float> inputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                       -8.0f, 12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, 11.0f };

    std::vector<float> expectedOutputValues = { 8.616844f, 11.543396f, 9.721111f, 7.632169f, 9.8234415f, 9.367497f };

    Pooling2dTest<float>(tflite::BuiltinOperator_L2_POOL_2D,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         ::tflite::Padding_VALID,
                         1,
                         1,
                         2,
                         2,
                         ::tflite::ActivationFunctionType_RELU);
}

void L2Pool2dFP32Relu6Test(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 4, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    std::vector<float> inputValues = { -5.0f, 8.0f, -10.0f, 7.0f,
                                       -8.0f, 12.0f, -15.0f, 2.0f,
                                       3.0f, -4.0f, -1.0f, 11.0f };

    std::vector<float> expectedOutputValues = { 5.0f, 6.0f, 3.0f, 1.0f };

    Pooling2dTest<float>(tflite::BuiltinOperator_L2_POOL_2D,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         ::tflite::Padding_SAME,
                         2,
                         2,
                         1,
                         1,
                         ::tflite::ActivationFunctionType_RELU6);
}

TEST_SUITE("Pooling2dTests")
{

TEST_CASE ("MaxPooling2d_FP32_PaddingValid_Test")
{
    MaxPool2dFP32PaddingValidTest();
}

TEST_CASE ("MaxPooling2d_Int8_PaddingValid_Test")
{
    MaxPool2dInt8PaddingValidTest();
}

TEST_CASE ("MaxPooling2d_FP32_PaddingSame_Test")
{
    MaxPool2dFP32PaddingSameTest();
}

TEST_CASE ("MaxPooling2d_Int8_PaddingSame_Test")
{
    MaxPool2dInt8PaddingSameTest();
}

TEST_CASE ("MaxPooling2d_FP32_Relu_Test")
{
    MaxPool2dFP32ReluTest();
}

TEST_CASE ("MaxPooling2d_Int8_Relu_Test")
{
    MaxPool2dInt8ReluTest();
}

TEST_CASE ("MaxPooling2d_FP32_Relu6_Test")
{
    MaxPool2dFP32Relu6Test();
}

TEST_CASE ("MaxPooling2d_Int8_Relu6_Test")
{
    MaxPool2dInt8Relu6Test();
}

TEST_CASE ("MaxPooling2d_Uint8_PaddingSame_Test")
{
    MaxPool2dUint8PaddingSameTest();
}

TEST_CASE ("MaxPooling2d_Uint8_Relu_Test")
{
    MaxPool2dUint8ReluTest();
}

TEST_CASE ("MaxPooling2d_Int16_PaddingSame_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MaxPool2dInt16PaddingSameTest(backends);
}

TEST_CASE ("MaxPooling2d_Int16_Relu_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MaxPool2dInt16ReluTest(backends);
}

TEST_CASE ("AveragePooling2d_FP32_PaddingValid_Test")
{
    AveragePool2dFP32PaddingValidTest();
}

TEST_CASE ("AveragePooling2d_Int8_PaddingValid_Test")
{
    AveragePool2dInt8PaddingValidTest();
}

TEST_CASE ("AveragePooling2d_FP32_PaddingSame_Test")
{
    AveragePool2dFP32PaddingSameTest();
}

TEST_CASE ("AveragePooling2d_Int8_PaddingSame_Test")
{
    AveragePool2dInt8PaddingSameTest();
}

TEST_CASE ("AveragePooling2d_FP32_Relu_Test")
{
    AveragePool2dFP32ReluTest();
}

TEST_CASE ("AveragePooling2d_FP32_Relu6_Test")
{
    AveragePool2dFP32Relu6Test();
}

TEST_CASE ("AveragePooling2d_Int8_Relu_Test")
{
    AveragePool2dInt8ReluTest();
}

TEST_CASE ("AveragePooling2d_Int8_Relu6_Test")
{
    AveragePool2dInt8Relu6Test();
}

TEST_CASE ("AveragePooling2d_Uint8_PaddingSame_Test")
{
    AveragePool2dUint8PaddingSameTest();
}

TEST_CASE ("AveragePooling2d_Uint8_Relu_Test")
{
    AveragePool2dUint8ReluTest();
}

TEST_CASE ("AveragePooling2d_Int16_PaddingSame_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    AveragePool2dInt16PaddingSameTest(backends);
}

TEST_CASE ("AveragePooling2d_Int16_Relu_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    AveragePool2dInt16ReluTest(backends);
}

TEST_CASE ("L2Pooling2d_FP32_PaddingValid_Test")
{
    L2Pool2dFP32PaddingValidTest();
}

TEST_CASE ("L2Pooling2d_FP32_PaddingSame_Test")
{
    L2Pool2dFP32PaddingSameTest();
}

TEST_CASE ("L2Pooling2d_FP32_Relu_Test")
{
    L2Pool2dFP32ReluTest();
}

TEST_CASE ("L2Pooling2d_FP32_Relu6_Test")
{
    L2Pool2dFP32Relu6Test();
}

} // TEST_SUITE("Pooling2dTests")

} // namespace armnnDelegate