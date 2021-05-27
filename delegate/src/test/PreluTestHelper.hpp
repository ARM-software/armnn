//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

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
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));

    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector(
        reinterpret_cast<const uint8_t *>(alphaData.data()), sizeof(float) * alphaData.size())));

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
                                    0,
                                    flatBufferBuilder.CreateString("input"),
                                    quantizationParameters);

    auto alphaTensor = CreateTensor(flatBufferBuilder,
                                    flatBufferBuilder.CreateVector<int32_t>(alphaShape.data(),
                                                                          alphaShape.size()),
                                    tensorType,
                                    1,
                                    flatBufferBuilder.CreateString("alpha"),
                                    quantizationParameters);

    auto outputTensor = CreateTensor(flatBufferBuilder,
                                     flatBufferBuilder.CreateVector<int32_t>(outputShape.data(),
                                                                           outputShape.size()),
                                     tensorType,
                                     0,
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

    flatBufferBuilder.Finish(flatbufferModel);

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
    using namespace tflite;

    std::vector<char> modelBuffer = CreatePreluTfLiteModel(preluOperatorCode,
                                                           tensorType,
                                                           inputShape,
                                                           alphaShape,
                                                           outputShape,
                                                           alphaData,
                                                           alphaIsConstant);

    const Model* tfLiteModel = GetModel(modelBuffer.data());

    CHECK(tfLiteModel != nullptr);

    std::unique_ptr<Interpreter> armnnDelegateInterpreter;

    CHECK(InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
          (&armnnDelegateInterpreter) == kTfLiteOk);
    CHECK(armnnDelegateInterpreter != nullptr);
    CHECK(armnnDelegateInterpreter->AllocateTensors() == kTfLiteOk);

    std::unique_ptr<Interpreter> tfLiteInterpreter;

    CHECK(InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
          (&tfLiteInterpreter) == kTfLiteOk);
    CHECK(tfLiteInterpreter != nullptr);
    CHECK(tfLiteInterpreter->AllocateTensors() == kTfLiteOk);

    // Create the ArmNN Delegate
    armnnDelegate::DelegateOptions delegateOptions(backends);

    std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
        theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                         armnnDelegate::TfLiteArmnnDelegateDelete);
    CHECK(theArmnnDelegate != nullptr);

    // Modify armnnDelegateInterpreter to use armnnDelegate
    CHECK(armnnDelegateInterpreter->ModifyGraphWithDelegate(theArmnnDelegate.get()) == kTfLiteOk);

    // Set input data
    armnnDelegate::FillInput<float>(tfLiteInterpreter, 0, inputData);
    armnnDelegate::FillInput<float>(armnnDelegateInterpreter, 0, inputData);

    // Set alpha data if not constant
    if (!alphaIsConstant) {
        armnnDelegate::FillInput<float>(tfLiteInterpreter, 1, alphaData);
        armnnDelegate::FillInput<float>(armnnDelegateInterpreter, 1, alphaData);
    }

    // Run EnqueueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    // Compare output data
    auto tfLiteDelegateOutputId = tfLiteInterpreter->outputs()[0];

    auto tfLiteDelegateOutputData = tfLiteInterpreter->typed_tensor<float>(tfLiteDelegateOutputId);

    auto armnnDelegateOutputId = armnnDelegateInterpreter->outputs()[0];
    auto armnnDelegateOutputData = armnnDelegateInterpreter->typed_tensor<float>(armnnDelegateOutputId);

    for (size_t i = 0; i < expectedOutput.size(); i++)
    {
        CHECK(expectedOutput[i] == armnnDelegateOutputData[i]);
        CHECK(tfLiteDelegateOutputData[i] == expectedOutput[i]);
        CHECK(tfLiteDelegateOutputData[i] == armnnDelegateOutputData[i]);
    }
}
} // anonymous namespace