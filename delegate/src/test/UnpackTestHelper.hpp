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

std::vector<char> CreateUnpackTfLiteModel(tflite::BuiltinOperator unpackOperatorCode,
                                          tflite::TensorType tensorType,
                                          std::vector<int32_t>& inputTensorShape,
                                          const std::vector <int32_t>& outputTensorShape,
                                          const int32_t outputTensorNum,
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

    const std::vector<int32_t> operatorInputs{ 0 };
    std::vector<int32_t> operatorOutputs{};
    const std::vector<int> subgraphInputs{ 0 };
    std::vector<int> subgraphOutputs{};

    std::vector<flatbuffers::Offset<Tensor>> tensors(outputTensorNum + 1);

    // Create input tensor
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                                                                      tensorType,
                                                                      0,
                                                                      flatBufferBuilder.CreateString("input"),
                                                                      quantizationParameters);

    for (int i = 0; i < outputTensorNum; ++i)
    {
        tensors[i + 1] = CreateTensor(flatBufferBuilder,
                                  flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                          outputTensorShape.size()),
                                  tensorType,
                                  0,
                                  flatBufferBuilder.CreateString("output" + std::to_string(i)),
                                  quantizationParameters);

        operatorOutputs.push_back(i + 1);
        subgraphOutputs.push_back(i + 1);
    }

    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_UnpackOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions =
        CreateUnpackOptions(flatBufferBuilder, outputTensorNum, axis).Union();

    flatbuffers::Offset <Operator> unpackOperator =
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
                       flatBufferBuilder.CreateVector(&unpackOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Unpack Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, unpackOperatorCode);

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
void UnpackTest(tflite::BuiltinOperator unpackOperatorCode,
              tflite::TensorType tensorType,
              std::vector<armnn::BackendId>& backends,
              std::vector<int32_t>& inputShape,
              std::vector<int32_t>& expectedOutputShape,
              std::vector<T>& inputValues,
              std::vector<std::vector<T>>& expectedOutputValues,
              unsigned int axis = 0,
              float quantScale = 1.0f,
              int quantOffset  = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateUnpackTfLiteModel(unpackOperatorCode,
                                                            tensorType,
                                                            inputShape,
                                                            expectedOutputShape,
                                                            expectedOutputValues.size(),
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

    // Set input data
    armnnDelegate::FillInput<T>(tfLiteInterpreter, 0, inputValues);
    armnnDelegate::FillInput<T>(armnnDelegateInterpreter, 0, inputValues);


    // Run EnqueueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    // Compare output data
    for (unsigned int i = 0; i < expectedOutputValues.size(); ++i)
    {
        armnnDelegate::CompareOutputData<T>(tfLiteInterpreter,
                                            armnnDelegateInterpreter,
                                            expectedOutputShape,
                                            expectedOutputValues[i],
                                            i);
    }

    armnnDelegateInterpreter.reset(nullptr);
}

} // anonymous namespace