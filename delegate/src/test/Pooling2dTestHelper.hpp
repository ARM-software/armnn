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

std::vector<char> CreatePooling2dTfLiteModel(
    tflite::BuiltinOperator poolingOperatorCode,
    tflite::TensorType tensorType,
    const std::vector <int32_t>& inputTensorShape,
    const std::vector <int32_t>& outputTensorShape,
    tflite::Padding padding = tflite::Padding_SAME,
    int32_t strideWidth = 0,
    int32_t strideHeight = 0,
    int32_t filterWidth = 0,
    int32_t filterHeight = 0,
    tflite::ActivationFunctionType fusedActivation = tflite::ActivationFunctionType_NONE,
    float quantScale = 1.0f,
    int quantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));

    auto quantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({ quantScale }),
                                     flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));

    std::array<flatbuffers::Offset<Tensor>, 2> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);

    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_Pool2DOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreatePool2DOptions(flatBufferBuilder,
                                                                           padding,
                                                                           strideWidth,
                                                                           strideHeight,
                                                                           filterWidth,
                                                                           filterHeight,
                                                                           fusedActivation).Union();

    const std::vector<int32_t> operatorInputs{0};
    const std::vector<int32_t> operatorOutputs{1};
    flatbuffers::Offset <Operator> poolingOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       operatorBuiltinOptionsType,
                       operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{0};
    const std::vector<int> subgraphOutputs{1};
    flatbuffers::Offset <SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&poolingOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Pooling2d Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, poolingOperatorCode);

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
void Pooling2dTest(tflite::BuiltinOperator poolingOperatorCode,
                   tflite::TensorType tensorType,
                   std::vector<armnn::BackendId>& backends,
                   std::vector<int32_t>& inputShape,
                   std::vector<int32_t>& outputShape,
                   std::vector<T>& inputValues,
                   std::vector<T>& expectedOutputValues,
                   tflite::Padding padding = tflite::Padding_SAME,
                   int32_t strideWidth = 0,
                   int32_t strideHeight = 0,
                   int32_t filterWidth = 0,
                   int32_t filterHeight = 0,
                   tflite::ActivationFunctionType fusedActivation = tflite::ActivationFunctionType_NONE,
                   float quantScale = 1.0f,
                   int quantOffset  = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreatePooling2dTfLiteModel(poolingOperatorCode,
                                                               tensorType,
                                                               inputShape,
                                                               outputShape,
                                                               padding,
                                                               strideWidth,
                                                               strideHeight,
                                                               filterWidth,
                                                               filterHeight,
                                                               fusedActivation,
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
    auto tfLiteDelegateInputId = tfLiteInterpreter->inputs()[0];
    auto tfLiteDelegateInputData = tfLiteInterpreter->typed_tensor<T>(tfLiteDelegateInputId);
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        tfLiteDelegateInputData[i] = inputValues[i];
    }

    auto armnnDelegateInputId = armnnDelegateInterpreter->inputs()[0];
    auto armnnDelegateInputData = armnnDelegateInterpreter->typed_tensor<T>(armnnDelegateInputId);
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        armnnDelegateInputData[i] = inputValues[i];
    }

    // Run EnqueueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    armnnDelegate::CompareOutputData(tfLiteInterpreter, armnnDelegateInterpreter, outputShape, expectedOutputValues);
}

} // anonymous namespace




