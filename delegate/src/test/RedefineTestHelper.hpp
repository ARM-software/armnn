//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
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

std::vector<char> CreateRedefineTfLiteModel(
    tflite::BuiltinOperator redefineOperatorCode,
    tflite::TensorType tensorType,
    const std::vector<int32_t>& inputTensorShape,
    const std::vector<int32_t>& outputTensorShape,
    const std::vector<int32_t>& targetShape,
    bool useOption = true,
    float quantScale = 1.0f,
    int quantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));

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
                                    0,
                                    flatBufferBuilder.CreateString("input"),
                                    quantizationParameters);

    auto outputTensor = CreateTensor(flatBufferBuilder,
                                     flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                             outputTensorShape.size()),
                                     tensorType,
                                     1,
                                     flatBufferBuilder.CreateString("output"),
                                     quantizationParameters);

    std::vector<flatbuffers::Offset<Tensor>> tensors;
    std::vector<int32_t> operatorInputs;
    std::vector<int> subgraphInputs;
    flatbuffers::Offset<void> operatorBuiltinOptions;

    if (useOption)
    {
        tensors = { inputTensor, outputTensor};
        operatorInputs = {0};
        subgraphInputs = {0};
        operatorBuiltinOptions = CreateReshapeOptions(
            flatBufferBuilder,
            flatBufferBuilder.CreateVector(targetShape.data(), targetShape.size())).Union();
    }
    else
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(targetShape.data()),
                                                        sizeof(int32_t) * targetShape.size())));
        int32_t size = static_cast<int32_t>(targetShape.size());
        auto shapeTensor = CreateTensor(flatBufferBuilder,
                                        flatBufferBuilder.CreateVector<int32_t>( { size } ),
                                        tflite::TensorType_INT32,
                                        2,
                                        flatBufferBuilder.CreateString("shape"));
        tensors = { inputTensor, outputTensor, shapeTensor };
        operatorInputs = {0, 2};
        subgraphInputs = {0, 2};
        operatorBuiltinOptions = CreateReshapeOptions(flatBufferBuilder).Union();
    }

    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_ReshapeOptions;

    const std::vector<int32_t> operatorOutputs{1};
    flatbuffers::Offset <Operator> redefineOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       operatorBuiltinOptionsType,
                       operatorBuiltinOptions);

    const std::vector<int> subgraphOutputs{1};
    flatbuffers::Offset <SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&redefineOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Reshape Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         redefineOperatorCode);

    flatbuffers::Offset <Model> flatbufferModel =
        CreateModel(flatBufferBuilder,
                    TFLITE_SCHEMA_VERSION,
                    flatBufferBuilder.CreateVector(&operatorCode, 1),
                    flatBufferBuilder.CreateVector(&subgraph, 1),
                    modelDescription,
                    flatBufferBuilder.CreateVector(buffers.data(), buffers.size()));

    flatBufferBuilder.Finish(flatbufferModel);

    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

template <typename T>
void RedefineTest(tflite::BuiltinOperator redefineOperatorCode,
                  tflite::TensorType tensorType,
                  const std::vector<armnn::BackendId>& backends,
                  const std::vector<int32_t>& inputShape,
                  std::vector<int32_t>& outputShape,
                  std::vector<T>& inputValues,
                  std::vector<T>& expectedOutputValues,
                  std::vector<int32_t>& targetShape,
                  bool useOption = true,
                  float quantScale = 1.0f,
                  int quantOffset  = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateRedefineTfLiteModel(redefineOperatorCode,
                                                              tensorType,
                                                              inputShape,
                                                              outputShape,
                                                              targetShape,
                                                              useOption,
                                                              quantScale,
                                                              quantOffset);

    const Model* tfLiteModel = GetModel(modelBuffer.data());
    CHECK(tfLiteModel != nullptr);
    // Create TfLite Interpreters
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
    armnnDelegate::FillInput<T>(tfLiteInterpreter, 0, inputValues);
    armnnDelegate::FillInput<T>(armnnDelegateInterpreter, 0, inputValues);

    // Run EnqueueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    armnnDelegate::CompareOutputData<T>(tfLiteInterpreter, armnnDelegateInterpreter, outputShape, expectedOutputValues);
}

} // anonymous namespace