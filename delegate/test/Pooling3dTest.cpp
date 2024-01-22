//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Pooling3dTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

// Pool3D custom op was only added in tflite r2.6.
#if defined(ARMNN_POST_TFLITE_2_5)

void MaxPool3dFP32PaddingValidTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input and expected output data
    std::vector<int32_t> inputShape = { 1, 2, 3, 4, 1 };
    std::vector<int32_t> outputShape = { 1, 1, 2, 3, 1 };

    std::vector<float> inputValues = { 1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6 };
    std::vector<float> expectedOutputValues = { 6, 6, 4 };

    // poolType string required to create the correct pooling operator
    // Padding type required to create the padding in custom options
    std::string poolType = "kMax";
    TfLitePadding padding = kTfLitePaddingValid;

    Pooling3dTest<float>(poolType,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         padding,
                         1,
                         1,
                         1,
                         2,
                         2,
                         2);
}

void MaxPool3dFP32PaddingSameTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data and expected output data
    std::vector<int32_t> inputShape = { 1, 2, 3, 4, 1 };
    std::vector<int32_t> outputShape = { 1, 2, 3, 4, 1 };

    std::vector<float> inputValues = { 1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6 };
    std::vector<float> expectedOutputValues = { 6, 6, 4, 4, 6, 6, 6, 6, 4, 5, 6, 6, 6, 6, 4, 4 };

    // poolType string required to create the correct pooling operator
    // Padding type required to create the padding in custom options
    std::string poolType = "kMax";
    TfLitePadding padding = kTfLitePaddingSame;

    Pooling3dTest<float>(poolType,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         padding,
                         1,
                         1,
                         1,
                         2,
                         2,
                         2);
}

void MaxPool3dFP32H1Test(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data and expected output data
    std::vector<int32_t> inputShape = { 1, 2, 3, 4, 1 };
    std::vector<int32_t> outputShape = { 1, 1, 3, 3, 1 };

    std::vector<float> inputValues = { 1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6 };
    std::vector<float> expectedOutputValues = { 2, 3 };

    // poolType string required to create the correct pooling operator
    // Padding type required to create the padding in custom options
    std::string poolType = "kMax";
    TfLitePadding padding = kTfLitePaddingValid;

    Pooling3dTest<float>(poolType,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         padding,
                         1,
                         1,
                         1,
                         2,
                         1,
                         2);
}

void MaxPool3dFP32Test(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data and expected output data
    std::vector<int32_t> inputShape = { 1, 2, 3, 4, 1 };
    std::vector<int32_t> outputShape = { 1, 2, 3, 4, 1 };

    std::vector<float> inputValues = { 1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6 };
    std::vector<float> expectedOutputValues = { 6, 6 };

    // poolType string required to create the correct pooling operator
    // Padding type required to create the padding in custom options
    std::string poolType = "kMax";
    TfLitePadding padding = kTfLitePaddingUnknown;

    Pooling3dTest<float>(poolType,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         padding,
                         1,
                         1,
                         1,
                         2,
                         2,
                         2);
}

void AveragePool3dFP32PaddingValidTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data and expected output data.
    std::vector<int32_t> inputShape = { 1, 2, 3, 4, 1 };
    std::vector<int32_t> outputShape = { 1, 1, 2, 3, 1 };

    std::vector<float> inputValues = { 1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6 };
    std::vector<float> expectedOutputValues = { 3.5, 3, 2.5 };

    // poolType string required to create the correct pooling operator
    // Padding type required to create the padding in custom options
    std::string poolType = "kAverage";
    TfLitePadding padding = kTfLitePaddingValid;

    Pooling3dTest<float>(poolType,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         padding,
                         1,
                         1,
                         1,
                         2,
                         2,
                         2);
}

void AveragePool3dFP32PaddingSameTest(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data and expected output data
    std::vector<int32_t> inputShape = { 4, 2, 3, 1, 1 };
    std::vector<int32_t> outputShape = { 4, 2, 3, 1, 1 };

    std::vector<float> inputValues = { 1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6 };
    std::vector<float> expectedOutputValues = { 3, 4, 4.5, 4.5, 5.5, 6, 3, 4, 4.5, 4.5, 5.5, 6, 3, 4, 4.5, 4.5 };

    // poolType string required to create the correct pooling operator
    // Padding type required to create the padding in custom options
    std::string poolType = "kAverage";
    TfLitePadding padding = kTfLitePaddingSame;

    Pooling3dTest<float>(poolType,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         padding,
                         1,
                         1,
                         1,
                         2,
                         2,
                         2);
}

void AveragePool3dFP32H1Test(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data and expected output data
    std::vector<int32_t> inputShape = { 1, 2, 3, 4, 1 };
    std::vector<int32_t> outputShape = { 1, 1, 2, 2, 1 };

    std::vector<float> inputValues = { 1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6 };
    std::vector<float> expectedOutputValues = { 1.5, 3.5 };

    // poolType string required to create the correct pooling operator
    // Padding type required to create the padding in custom options
    std::string poolType = "kAverage";
    TfLitePadding padding = kTfLitePaddingUnknown;

    Pooling3dTest<float>(poolType,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         padding,
                         2,
                         2,
                         2,
                         2,
                         1,
                         2);
}

void AveragePool3dFP32Test(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data and expected output data
    std::vector<int32_t> inputShape = { 4, 3, 2, 1, 1 };
    std::vector<int32_t> outputShape = { 1, 2, 2, 4, 1 };

    std::vector<float> inputValues = { 1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6,
                                       1, 2, 3, 4, 5, 6 };
    std::vector<float> expectedOutputValues = { 3.125, 4.25 };

    // poolType string required to create the correct pooling operator
    // Padding type required to create the padding in custom options
    std::string poolType = "kMax";
    TfLitePadding padding = kTfLitePaddingUnknown;

    Pooling3dTest<float>(poolType,
                         ::tflite::TensorType_FLOAT32,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         backends,
                         padding,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2);
}

TEST_SUITE("Pooling3dTests")
{
TEST_CASE ("MaxPooling3d_FP32_Test")
{
    MaxPool3dFP32Test();
}

TEST_CASE ("MaxPooling3d_FP32_PaddingValid_Test")
{
    MaxPool3dFP32PaddingValidTest();
}

TEST_CASE ("MaxPooling3d_FP32_PaddingSame_Test")
{
    MaxPool3dFP32PaddingSameTest();
}

TEST_CASE ("MaxPooling3d_FP32_H1_Test")
{
    MaxPool3dFP32H1Test();
}

TEST_CASE ("AveragePooling3d_FP32_PaddingValid_Test")
{
    AveragePool3dFP32PaddingValidTest();
}

TEST_CASE ("AveragePooling3d_FP32_PaddingSame_Test")
{
    AveragePool3dFP32PaddingSameTest();
}

TEST_CASE ("AveragePooling3d_FP32_H1_Test")
{
    AveragePool3dFP32H1Test();
}

} // TEST_SUITE("Pooling3dTests")

#endif

}