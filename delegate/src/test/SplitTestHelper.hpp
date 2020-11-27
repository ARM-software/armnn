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

std::vector<char> CreateSplitTfLiteModel(tflite::TensorType tensorType,
                                         std::vector<int32_t>& axisTensorShape,
                                         std::vector<int32_t>& inputTensorShape,
                                         const std::vector<std::vector<int32_t>>& outputTensorShapes,
                                         std::vector<int32_t>& axisData,
                                         const int32_t numSplits,
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

    std::array<flatbuffers::Offset<Tensor>, 4> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(axisTensorShape.data(),
                                                                      axisTensorShape.size()),
                              ::tflite::TensorType_INT32,
                              1,
                              flatBufferBuilder.CreateString("axis"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);

    // Create output tensor
    for (unsigned int i = 0; i < outputTensorShapes.size(); ++i)
    {
        tensors[i + 2] = CreateTensor(flatBufferBuilder,
                                  flatBufferBuilder.CreateVector<int32_t>(outputTensorShapes[i].data(),
                                                                          outputTensorShapes[i].size()),
                                  tensorType,
                                  0,
                                  flatBufferBuilder.CreateString("output"),
                                  quantizationParameters);
    }

    // create operator. Mean uses ReducerOptions.
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_SplitOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateSplitOptions(flatBufferBuilder, numSplits).Union();

    const std::vector<int> operatorInputs{ {0, 1} };
    const std::vector<int> operatorOutputs{ {2, 3} };
    flatbuffers::Offset <Operator> controlOperator =
            CreateOperator(flatBufferBuilder,
                           0,
                           flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                           operatorBuiltinOptionsType,
                           operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{ {0, 1} };
    const std::vector<int> subgraphOutputs{ {2, 3} };
    flatbuffers::Offset <SubGraph> subgraph =
            CreateSubGraph(flatBufferBuilder,
                           flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                           flatBufferBuilder.CreateVector(&controlOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: SPLIT Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, BuiltinOperator_SPLIT);

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
void SplitTest(tflite::TensorType tensorType,
               std::vector<armnn::BackendId>& backends,
               std::vector<int32_t>& axisTensorShape,
               std::vector<int32_t>& inputTensorShape,
               std::vector<std::vector<int32_t>>& outputTensorShapes,
               std::vector<int32_t>& axisData,
               std::vector<T>& inputValues,
               std::vector<std::vector<T>>& expectedOutputValues,
               const int32_t numSplits,
               float quantScale = 1.0f,
               int quantOffset  = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateSplitTfLiteModel(tensorType,
                                                           axisTensorShape,
                                                           inputTensorShape,
                                                           outputTensorShapes,
                                                           axisData,
                                                           numSplits,
                                                           quantScale,
                                                           quantOffset);
    const Model* tfLiteModel = GetModel(modelBuffer.data());

    // Create TfLite Interpreters
    std::unique_ptr<Interpreter> armnnDelegate;
    CHECK(InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
              (&armnnDelegate) == kTfLiteOk);
    CHECK(armnnDelegate != nullptr);
    CHECK(armnnDelegate->AllocateTensors() == kTfLiteOk);

    std::unique_ptr<Interpreter> tfLiteDelegate;
    CHECK(InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
              (&tfLiteDelegate) == kTfLiteOk);
    CHECK(tfLiteDelegate != nullptr);
    CHECK(tfLiteDelegate->AllocateTensors() == kTfLiteOk);

    // Create the ArmNN Delegate
    armnnDelegate::DelegateOptions delegateOptions(backends);
    std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
    theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                     armnnDelegate::TfLiteArmnnDelegateDelete);
    CHECK(theArmnnDelegate != nullptr);

    // Modify armnnDelegateInterpreter to use armnnDelegate
    CHECK(armnnDelegate->ModifyGraphWithDelegate(theArmnnDelegate.get()) == kTfLiteOk);

    // Set input data
    armnnDelegate::FillInput<T>(tfLiteDelegate, 1, inputValues);
    armnnDelegate::FillInput<T>(armnnDelegate, 1, inputValues);

    // Run EnqueWorkload
    CHECK(tfLiteDelegate->Invoke() == kTfLiteOk);
    CHECK(armnnDelegate->Invoke() == kTfLiteOk);

    // Compare output data
    for (unsigned int i = 0; i < expectedOutputValues.size(); ++i)
    {
        armnnDelegate::CompareOutputData<T>(tfLiteDelegate,
                                            armnnDelegate,
                                            outputTensorShapes[i],
                                            expectedOutputValues[i],
                                            i);
    }

    tfLiteDelegate.reset(nullptr);
    armnnDelegate.reset(nullptr);
} // End of SPLIT Test

std::vector<char> CreateSplitVTfLiteModel(tflite::TensorType tensorType,
                                          std::vector<int32_t>& inputTensorShape,
                                          std::vector<int32_t>& splitsTensorShape,
                                          std::vector<int32_t>& axisTensorShape,
                                          const std::vector<std::vector<int32_t>>& outputTensorShapes,
                                          std::vector<int32_t>& splitsData,
                                          std::vector<int32_t>& axisData,
                                          const int32_t numSplits,
                                          float quantScale = 1.0f,
                                          int quantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::array<flatbuffers::Offset<tflite::Buffer>, 3> buffers;
    buffers[0] = CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({}));
    buffers[1] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(splitsData.data()),
                                                             sizeof(int32_t) * splitsData.size()));
    buffers[2] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(axisData.data()),
                                                             sizeof(int32_t) * axisData.size()));

    auto quantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({ quantScale }),
                                     flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));

    std::array<flatbuffers::Offset<Tensor>, 5> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(splitsTensorShape.data(),
                                                                      splitsTensorShape.size()),
                              ::tflite::TensorType_INT32,
                              1,
                              flatBufferBuilder.CreateString("splits"),
                              quantizationParameters);
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(axisTensorShape.data(),
                                                                      axisTensorShape.size()),
                              ::tflite::TensorType_INT32,
                              2,
                              flatBufferBuilder.CreateString("axis"),
                              quantizationParameters);

    // Create output tensor
    for (unsigned int i = 0; i < outputTensorShapes.size(); ++i)
    {
        tensors[i + 3] = CreateTensor(flatBufferBuilder,
                                      flatBufferBuilder.CreateVector<int32_t>(outputTensorShapes[i].data(),
                                                                              outputTensorShapes[i].size()),
                                      tensorType,
                                      0,
                                      flatBufferBuilder.CreateString("output"),
                                      quantizationParameters);
    }

    // create operator. Mean uses ReducerOptions.
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_SplitVOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateSplitVOptions(flatBufferBuilder, numSplits).Union();

    const std::vector<int> operatorInputs{ {0, 1, 2} };
    const std::vector<int> operatorOutputs{ {3, 4} };
    flatbuffers::Offset <Operator> controlOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       operatorBuiltinOptionsType,
                       operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{ {0, 1, 2} };
    const std::vector<int> subgraphOutputs{ {3, 4} };
    flatbuffers::Offset <SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&controlOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: SPLIT_V Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, BuiltinOperator_SPLIT_V);

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
void SplitVTest(tflite::TensorType tensorType,
                std::vector<armnn::BackendId>& backends,
                std::vector<int32_t>& inputTensorShape,
                std::vector<int32_t>& splitsTensorShape,
                std::vector<int32_t>& axisTensorShape,
                std::vector<std::vector<int32_t>>& outputTensorShapes,
                std::vector<T>& inputValues,
                std::vector<int32_t>& splitsData,
                std::vector<int32_t>& axisData,
                std::vector<std::vector<T>>& expectedOutputValues,
                const int32_t numSplits,
                float quantScale = 1.0f,
                int quantOffset  = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateSplitVTfLiteModel(tensorType,
                                                            inputTensorShape,
                                                            splitsTensorShape,
                                                            axisTensorShape,
                                                            outputTensorShapes,
                                                            splitsData,
                                                            axisData,
                                                            numSplits,
                                                            quantScale,
                                                            quantOffset);
    const Model* tfLiteModel = GetModel(modelBuffer.data());

    // Create TfLite Interpreters
    std::unique_ptr<Interpreter> armnnDelegate;
    CHECK(InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
                  (&armnnDelegate) == kTfLiteOk);
    CHECK(armnnDelegate != nullptr);
    CHECK(armnnDelegate->AllocateTensors() == kTfLiteOk);

    std::unique_ptr<Interpreter> tfLiteDelegate;
    CHECK(InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
                  (&tfLiteDelegate) == kTfLiteOk);
    CHECK(tfLiteDelegate != nullptr);
    CHECK(tfLiteDelegate->AllocateTensors() == kTfLiteOk);

    // Create the ArmNN Delegate
    armnnDelegate::DelegateOptions delegateOptions(backends);
    std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
            theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                             armnnDelegate::TfLiteArmnnDelegateDelete);
    CHECK(theArmnnDelegate != nullptr);

    // Modify armnnDelegateInterpreter to use armnnDelegate
    CHECK(armnnDelegate->ModifyGraphWithDelegate(theArmnnDelegate.get()) == kTfLiteOk);

    // Set input data
    armnnDelegate::FillInput<T>(tfLiteDelegate, 0, inputValues);
    armnnDelegate::FillInput<T>(armnnDelegate, 0, inputValues);

    // Run EnqueWorkload
    CHECK(tfLiteDelegate->Invoke() == kTfLiteOk);
    CHECK(armnnDelegate->Invoke() == kTfLiteOk);

    // Compare output data
    for (unsigned int i = 0; i < expectedOutputValues.size(); ++i)
    {
        armnnDelegate::CompareOutputData<T>(tfLiteDelegate,
                                            armnnDelegate,
                                            outputTensorShapes[i],
                                            expectedOutputValues[i],
                                            i);
    }

    tfLiteDelegate.reset(nullptr);
    armnnDelegate.reset(nullptr);
} // End of SPLIT_V Test

} // anonymous namespace