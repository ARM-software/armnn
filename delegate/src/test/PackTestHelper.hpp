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

#include <string>

namespace
{

std::vector<char> CreatePackTfLiteModel(tflite::BuiltinOperator packOperatorCode,
                                        tflite::TensorType tensorType,
                                        std::vector<int32_t>& inputTensorShape,
                                        const std::vector <int32_t>& outputTensorShape,
                                        const int32_t inputTensorNum,
                                        unsigned int axis = 0,
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

    std::vector<int32_t> operatorInputs{};
    const std::vector<int32_t> operatorOutputs{inputTensorNum};
    std::vector<int> subgraphInputs{};
    const std::vector<int> subgraphOutputs{inputTensorNum};

    std::vector<flatbuffers::Offset<Tensor>> tensors(inputTensorNum + 1);
    for (int i = 0; i < inputTensorNum; ++i)
    {
        tensors[i] = CreateTensor(flatBufferBuilder,
                                  flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                          inputTensorShape.size()),
                                  tensorType,
                                  0,
                                  flatBufferBuilder.CreateString("input" + std::to_string(i)),
                                  quantizationParameters);

        // Add number of inputs to vector.
        operatorInputs.push_back(i);
        subgraphInputs.push_back(i);
    }

    // Create output tensor
    tensors[inputTensorNum] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_PackOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions =
            CreatePackOptions(flatBufferBuilder, inputTensorNum, axis).Union();

    flatbuffers::Offset <Operator> packOperator =
            CreateOperator(flatBufferBuilder,
                           0,
                           flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                           operatorBuiltinOptionsType,
                           operatorBuiltinOptions);

    flatbuffers::Offset <SubGraph> subgraph =
            CreateSubGraph(flatBufferBuilder,
                           flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                           flatBufferBuilder.CreateVector(&packOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: Pack Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, packOperatorCode);

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
void PackTest(tflite::BuiltinOperator packOperatorCode,
              tflite::TensorType tensorType,
              std::vector<armnn::BackendId>& backends,
              std::vector<int32_t>& inputShape,
              std::vector<int32_t>& expectedOutputShape,
              std::vector<std::vector<T>>& inputValues,
              std::vector<T>& expectedOutputValues,
              unsigned int axis = 0,
              float quantScale = 1.0f,
              int quantOffset  = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreatePackTfLiteModel(packOperatorCode,
                                                          tensorType,
                                                          inputShape,
                                                          expectedOutputShape,
                                                          inputValues.size(),
                                                          axis,
                                                          quantScale,
                                                          quantOffset);

    const Model* tfLiteModel = GetModel(modelBuffer.data());

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

    // Set input data for all input tensors.
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        // Get single input tensor and assign to interpreters.
        auto inputTensorValues = inputValues[i];
        armnnDelegate::FillInput<T>(tfLiteInterpreter, i, inputTensorValues);
        armnnDelegate::FillInput<T>(armnnDelegateInterpreter, i, inputTensorValues);
    }

    // Run EnqueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    // Compare output data
    armnnDelegate::CompareOutputData<T>(tfLiteInterpreter,
                                        armnnDelegateInterpreter,
                                        expectedOutputShape,
                                        expectedOutputValues);

    armnnDelegateInterpreter.reset(nullptr);
}

} // anonymous namespace