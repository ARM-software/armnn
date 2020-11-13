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

#include <string>

namespace
{

std::vector<char> CreateConcatTfLiteModel(tflite::BuiltinOperator controlOperatorCode,
                                          tflite::TensorType tensorType,
                                          std::vector<int32_t>& inputTensorShape,
                                          const std::vector <int32_t>& outputTensorShape,
                                          const int32_t inputTensorNum,
                                          int32_t axis = 0,
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
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_ConcatenationOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateConcatenationOptions(flatBufferBuilder, axis).Union();

    flatbuffers::Offset <Operator> controlOperator =
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
                           flatBufferBuilder.CreateVector(&controlOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: Concatenation Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, controlOperatorCode);

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

std::vector<char> CreateMeanTfLiteModel(tflite::BuiltinOperator controlOperatorCode,
                                        tflite::TensorType tensorType,
                                        std::vector<int32_t>& input0TensorShape,
                                        std::vector<int32_t>& input1TensorShape,
                                        const std::vector <int32_t>& outputTensorShape,
                                        std::vector<int32_t>& axisData,
                                        const bool keepDims,
                                        float quantScale = 1.0f,
                                        int quantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::array<flatbuffers::Offset<tflite::Buffer>, 2> buffers;
    buffers[0] = CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({}));
    buffers[1] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(axisData.data()),
                                                             sizeof(int32_t) * axisData.size()));

    auto quantizationParameters =
            CreateQuantizationParameters(flatBufferBuilder,
                                         0,
                                         0,
                                         flatBufferBuilder.CreateVector<float>({ quantScale }),
                                         flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));

    std::array<flatbuffers::Offset<Tensor>, 3> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(input0TensorShape.data(),
                                                                      input0TensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);

    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(input1TensorShape.data(),
                                                                      input1TensorShape.size()),
                              ::tflite::TensorType_INT32,
                              1,
                              flatBufferBuilder.CreateString("axis"),
                              quantizationParameters);

    // Create output tensor
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    // create operator. Mean uses ReducerOptions.
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_ReducerOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateReducerOptions(flatBufferBuilder, keepDims).Union();

    const std::vector<int> operatorInputs{ {0, 1} };
    const std::vector<int> operatorOutputs{ 2 };
    flatbuffers::Offset <Operator> controlOperator =
            CreateOperator(flatBufferBuilder,
                           0,
                           flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                           operatorBuiltinOptionsType,
                           operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{ {0, 1} };
    const std::vector<int> subgraphOutputs{ 2 };
    flatbuffers::Offset <SubGraph> subgraph =
            CreateSubGraph(flatBufferBuilder,
                           flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                           flatBufferBuilder.CreateVector(&controlOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: Mean Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, controlOperatorCode);

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
void ConcatenationTest(tflite::BuiltinOperator controlOperatorCode,
                       tflite::TensorType tensorType,
                       std::vector<armnn::BackendId>& backends,
                       std::vector<int32_t>& inputShapes,
                       std::vector<int32_t>& expectedOutputShape,
                       std::vector<std::vector<T>>& inputValues,
                       std::vector<T>& expectedOutputValues,
                       int32_t axis = 0,
                       float quantScale = 1.0f,
                       int quantOffset  = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateConcatTfLiteModel(controlOperatorCode,
                                                            tensorType,
                                                            inputShapes,
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

template <typename T>
void MeanTest(tflite::BuiltinOperator controlOperatorCode,
              tflite::TensorType tensorType,
              std::vector<armnn::BackendId>& backends,
              std::vector<int32_t>& input0Shape,
              std::vector<int32_t>& input1Shape,
              std::vector<int32_t>& expectedOutputShape,
              std::vector<T>& input0Values,
              std::vector<int32_t>& input1Values,
              std::vector<T>& expectedOutputValues,
              const bool keepDims,
              float quantScale = 1.0f,
              int quantOffset  = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateMeanTfLiteModel(controlOperatorCode,
                                                          tensorType,
                                                          input0Shape,
                                                          input1Shape,
                                                          expectedOutputShape,
                                                          input1Values,
                                                          keepDims,
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
    armnnDelegate::FillInput<T>(tfLiteInterpreter, 0, input0Values);
    armnnDelegate::FillInput<T>(armnnDelegateInterpreter, 0, input0Values);

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