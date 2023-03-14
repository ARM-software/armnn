//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Pooling3dTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <schema_generated.h>
#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

// Pool3D custom op was only added in tflite r2.6.
#if defined(ARMNN_POST_TFLITE_2_5)

void MaxPool3dFP32PaddingValidTest(std::vector<armnn::BackendId>& backends)
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
                         backends,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         padding,
                         1,
                         1,
                         1,
                         2,
                         2,
                         2);
}

void MaxPool3dFP32PaddingSameTest(std::vector<armnn::BackendId>& backends)
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
                         backends,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         padding,
                         1,
                         1,
                         1,
                         2,
                         2,
                         2);
}

void MaxPool3dFP32H1Test(std::vector<armnn::BackendId>& backends)
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
                         backends,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         padding,
                         1,
                         1,
                         1,
                         2,
                         1,
                         2);
}

void MaxPool3dFP32Test(std::vector<armnn::BackendId>& backends)
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
                         backends,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         padding,
                         1,
                         1,
                         1,
                         2,
                         2,
                         2);
}

void AveragePool3dFP32PaddingValidTest(std::vector<armnn::BackendId>& backends)
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
                         backends,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         padding,
                         1,
                         1,
                         1,
                         2,
                         2,
                         2);
}

void AveragePool3dFP32PaddingSameTest(std::vector<armnn::BackendId>& backends)
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
                         backends,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         padding,
                         1,
                         1,
                         1,
                         2,
                         2,
                         2);
}

void AveragePool3dFP32H1Test(std::vector<armnn::BackendId>& backends)
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
                         backends,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         padding,
                         2,
                         2,
                         2,
                         2,
                         1,
                         2);
}

void AveragePool3dFP32Test(std::vector<armnn::BackendId>& backends)
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
                         backends,
                         inputShape,
                         outputShape,
                         inputValues,
                         expectedOutputValues,
                         padding,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2);
}

TEST_SUITE("Pooling3d_GpuAccTests")
{

TEST_CASE ("MaxPooling3d_FP32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    MaxPool3dFP32Test(backends);
}

TEST_CASE ("MaxPooling3d_FP32_PaddingValid_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    MaxPool3dFP32PaddingValidTest(backends);
}

TEST_CASE ("MaxPooling3d_FP32_PaddingSame_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    MaxPool3dFP32PaddingSameTest(backends);
}

TEST_CASE ("MaxPooling3d_FP32_H1_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    MaxPool3dFP32H1Test(backends);
}

TEST_CASE ("AveragePooling3d_FP32_PaddingValid_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    AveragePool3dFP32PaddingValidTest(backends);
}

TEST_CASE ("AveragePooling3d_FP32_PaddingSame_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    AveragePool3dFP32PaddingSameTest(backends);
}

TEST_CASE ("AveragePooling3d_FP32_H1_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    AveragePool3dFP32H1Test(backends);
}

} // TEST_SUITE("Pooling3d_GpuAccTests")

TEST_SUITE("Pooling3d_CpuAccTests")
{

TEST_CASE ("MaxPooling3d_FP32_PaddingValid_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    MaxPool3dFP32PaddingValidTest(backends);
}

TEST_CASE ("MaxPooling3d_FP32_PaddingSame_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    MaxPool3dFP32PaddingSameTest(backends);
}

TEST_CASE ("MaxPooling3d_FP32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    MaxPool3dFP32Test(backends);
}

TEST_CASE ("MaxPooling3d_FP32_H1_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    MaxPool3dFP32H1Test(backends);
}

TEST_CASE ("AveragePooling3d_FP32_PaddingValid_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    AveragePool3dFP32PaddingValidTest(backends);
}

TEST_CASE ("AveragePooling3d_FP32_PaddingSame_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    AveragePool3dFP32PaddingSameTest(backends);
}

TEST_CASE ("AveragePooling3d_FP32_H1_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    AveragePool3dFP32H1Test(backends);
}

} // TEST_SUITE("Pooling3d_CpuAccTests")

TEST_SUITE("Pooling3d_CpuRefTests")
{
TEST_CASE ("MaxPooling3d_FP32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MaxPool3dFP32Test(backends);
}

TEST_CASE ("MaxPooling3d_FP32_PaddingValid_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MaxPool3dFP32PaddingValidTest(backends);
}

TEST_CASE ("MaxPooling3d_FP32_PaddingSame_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MaxPool3dFP32PaddingSameTest(backends);
}

TEST_CASE ("MaxPooling3d_FP32_H1_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MaxPool3dFP32H1Test(backends);
}

TEST_CASE ("AveragePooling3d_FP32_PaddingValid_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    AveragePool3dFP32PaddingValidTest(backends);
}

TEST_CASE ("AveragePooling3d_FP32_PaddingSame_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    AveragePool3dFP32PaddingSameTest(backends);
}

TEST_CASE ("AveragePooling3d_FP32_H1_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    AveragePool3dFP32H1Test(backends);
}

} // TEST_SUITE("Pooling3d_CpuRefTests")

#endif

}