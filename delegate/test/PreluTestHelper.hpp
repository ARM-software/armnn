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

std::vector<char> CreatePreluTfLiteModel(tflite::BuiltinOperator preluOperatorCode,
                                         tflite::TensorType tensorType,
                                         const std::vector<int32_t>& inputShape,
                                         const std::vector<int32_t>& alphaShape,
                                         const std::vector<int32_t>& outputShape,
                                         std::vector<float>& alphaData,
                                         bool alphaIsConstant)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector(
        reinterpret_cast<const uint8_t *>(alphaData.data()), sizeof(float) * alphaData.size())));
    buffers.push_back(CreateBuffer(flatBufferBuilder));


    auto quantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({ 1.0f }),
                                     flatBufferBuilder.CreateVector<int64_t>({ 0 }));

    auto inputTensor = CreateTensor(flatBufferBuilder,
                                    flatBufferBuilder.CreateVector<int32_t>(inputShape.data(),
                                                                          inputShape.size()),
                                    tensorType,
                                    1,
                                    flatBufferBuilder.CreateString("input"),
                                    quantizationParameters);

    auto alphaTensor = CreateTensor(flatBufferBuilder,
                                    flatBufferBuilder.CreateVector<int32_t>(alphaShape.data(),
                                                                          alphaShape.size()),
                                    tensorType,
                                    2,
                                    flatBufferBuilder.CreateString("alpha"),
                                    quantizationParameters);

    auto outputTensor = CreateTensor(flatBufferBuilder,
                                     flatBufferBuilder.CreateVector<int32_t>(outputShape.data(),
                                                                           outputShape.size()),
                                     tensorType,
                                     3,
                                     flatBufferBuilder.CreateString("output"),
                                     quantizationParameters);

    std::vector<flatbuffers::Offset<Tensor>> tensors = { inputTensor, alphaTensor, outputTensor };

    const std::vector<int> operatorInputs{0, 1};
    const std::vector<int> operatorOutputs{2};
    flatbuffers::Offset <Operator> preluOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()));

    std::vector<int> subgraphInputs{0};
    if (!alphaIsConstant)
    {
        subgraphInputs.push_back(1);
    }

    const std::vector<int> subgraphOutputs{2};
    flatbuffers::Offset <SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&preluOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Prelu Operator Model");
    flatbuffers::Offset <OperatorCode> opCode = CreateOperatorCode(flatBufferBuilder, preluOperatorCode);

    flatbuffers::Offset <Model> flatbufferModel =
        CreateModel(flatBufferBuilder,
                    TFLITE_SCHEMA_VERSION,
                    flatBufferBuilder.CreateVector(&opCode, 1),
                    flatBufferBuilder.CreateVector(&subgraph, 1),
                    modelDescription,
                    flatBufferBuilder.CreateVector(buffers.data(), buffers.size()));

    flatBufferBuilder.Finish(flatbufferModel, armnnDelegate::FILE_IDENTIFIER);

    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

void PreluTest(tflite::BuiltinOperator preluOperatorCode,
               tflite::TensorType tensorType,
               const std::vector<armnn::BackendId>& backends,
               const std::vector<int32_t>& inputShape,
               const std::vector<int32_t>& alphaShape,
               std::vector<int32_t>& outputShape,
               std::vector<float>& inputData,
               std::vector<float>& alphaData,
               std::vector<float>& expectedOutput,
               bool alphaIsConstant)
{
    using namespace delegateTestInterpreter;

    std::vector<char> modelBuffer = CreatePreluTfLiteModel(preluOperatorCode,
                                                           tensorType,
                                                           inputShape,
                                                           alphaShape,
                                                           outputShape,
                                                           alphaData,
                                                           alphaIsConstant);


    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);

    CHECK(armnnInterpreter.FillInputTensor<float>(inputData, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<float>(inputData, 0) == kTfLiteOk);

    // Set alpha data if not constant
    if (!alphaIsConstant)
    {
        CHECK(tfLiteInterpreter.FillInputTensor<float>(alphaData, 1) == kTfLiteOk);
        CHECK(armnnInterpreter.FillInputTensor<float>(alphaData, 1) == kTfLiteOk);
    }

    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<float>   tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<float>(0);

    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<float>   armnnOutputValues = armnnInterpreter.GetOutputResult<float>(0);

    armnnDelegate::CompareOutputData<float>(tfLiteOutputValues, armnnOutputValues, expectedOutput);

    // Don't compare shapes on dynamic output tests, as output shape gets cleared.
    if(!outputShape.empty())
    {
        std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);
        std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);
        armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, outputShape);
    }

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}
} // anonymous namespace