//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn_delegate.hpp>
#include <DelegateTestInterpreter.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/version.h>

#include <schema_generated.h>

#include <doctest/doctest.h>

namespace
{

std::vector<char> CreateNormalizationTfLiteModel(tflite::BuiltinOperator normalizationOperatorCode,
                                                 tflite::TensorType tensorType,
                                                 const std::vector<int32_t>& inputTensorShape,
                                                 const std::vector<int32_t>& outputTensorShape,
                                                 int32_t radius,
                                                 float bias,
                                                 float alpha,
                                                 float beta,
                                                 float quantScale = 1.0f,
                                                 int quantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    auto quantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({ quantScale }),
                                     flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));

    auto inputTensor = CreateTensor(flatBufferBuilder,
                                    flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                            inputTensorShape.size()),
                                    tensorType,
                                    1,
                                    flatBufferBuilder.CreateString("input"),
                                    quantizationParameters);

    auto outputTensor = CreateTensor(flatBufferBuilder,
                                     flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                             outputTensorShape.size()),
                                     tensorType,
                                     2,
                                     flatBufferBuilder.CreateString("output"),
                                     quantizationParameters);

    std::vector<flatbuffers::Offset<Tensor>> tensors = { inputTensor, outputTensor };

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));

    std::vector<int32_t> operatorInputs = { 0 };
    std::vector<int> subgraphInputs = { 0 };

    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_L2NormOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateL2NormOptions(flatBufferBuilder,
                                                                           tflite::ActivationFunctionType_NONE).Union();

    if (normalizationOperatorCode == tflite::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION)
    {
        operatorBuiltinOptionsType = BuiltinOptions_LocalResponseNormalizationOptions;
        operatorBuiltinOptions =
            CreateLocalResponseNormalizationOptions(flatBufferBuilder, radius, bias, alpha, beta).Union();
    }

    // create operator
    const std::vector<int32_t> operatorOutputs{ 1 };
    flatbuffers::Offset <Operator> normalizationOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       operatorBuiltinOptionsType,
                       operatorBuiltinOptions);

    const std::vector<int> subgraphOutputs{ 1 };
    flatbuffers::Offset <SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&normalizationOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Normalization Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         normalizationOperatorCode);

    flatbuffers::Offset <Model> flatbufferModel =
        CreateModel(flatBufferBuilder,
                    TFLITE_SCHEMA_VERSION,
                    flatBufferBuilder.CreateVector(&operatorCode, 1),
                    flatBufferBuilder.CreateVector(&subgraph, 1),
                    modelDescription,
                    flatBufferBuilder.CreateVector(buffers.data(), buffers.size()));

    flatBufferBuilder.Finish(flatbufferModel, armnnDelegate::FILE_IDENTIFIER);

    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

template <typename T>
void NormalizationTest(tflite::BuiltinOperator normalizationOperatorCode,
                       tflite::TensorType tensorType,
                       const std::vector<armnn::BackendId>& backends,
                       const std::vector<int32_t>& inputShape,
                       std::vector<int32_t>& outputShape,
                       std::vector<T>& inputValues,
                       std::vector<T>& expectedOutputValues,
                       int32_t radius = 0,
                       float bias = 0.f,
                       float alpha = 0.f,
                       float beta = 0.f,
                       float quantScale = 1.0f,
                       int quantOffset  = 0)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateNormalizationTfLiteModel(normalizationOperatorCode,
                                                                   tensorType,
                                                                   inputShape,
                                                                   outputShape,
                                                                   radius,
                                                                   bias,
                                                                   alpha,
                                                                   beta,
                                                                   quantScale,
                                                                   quantOffset);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, outputShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

void L2NormalizationTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape  { 1, 1, 1, 10 };
    std::vector<int32_t> outputShape { 1, 1, 1, 10 };

    std::vector<float> inputValues
    {
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f,
        6.0f,
        7.0f,
        8.0f,
        9.0f,
        10.0f
    };

    const float approxInvL2Norm = 0.050964719f;
    std::vector<float> expectedOutputValues
    {
        1.0f  * approxInvL2Norm,
        2.0f  * approxInvL2Norm,
        3.0f  * approxInvL2Norm,
        4.0f  * approxInvL2Norm,
        5.0f  * approxInvL2Norm,
        6.0f  * approxInvL2Norm,
        7.0f  * approxInvL2Norm,
        8.0f  * approxInvL2Norm,
        9.0f  * approxInvL2Norm,
        10.0f * approxInvL2Norm
    };

    NormalizationTest<float>(tflite::BuiltinOperator_L2_NORMALIZATION,
                             ::tflite::TensorType_FLOAT32,
                             backends,
                             inputShape,
                             outputShape,
                             inputValues,
                             expectedOutputValues);
}

void LocalResponseNormalizationTest(std::vector<armnn::BackendId>& backends,
                                    int32_t radius,
                                    float bias,
                                    float alpha,
                                    float beta)
{
    // Set input data
    std::vector<int32_t> inputShape  { 2, 2, 2, 1 };
    std::vector<int32_t> outputShape { 2, 2, 2, 1 };

    std::vector<float> inputValues
    {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    std::vector<float> expectedOutputValues
    {
        0.5f, 0.400000006f, 0.300000012f, 0.235294119f,
        0.192307696f, 0.16216217f, 0.140000001f, 0.123076923f
    };

    NormalizationTest<float>(tflite::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION,
                             ::tflite::TensorType_FLOAT32,
                             backends,
                             inputShape,
                             outputShape,
                             inputValues,
                             expectedOutputValues,
                             radius,
                             bias,
                             alpha,
                             beta);
}

} // anonymous namespace