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

std::vector<char> CreateGatherTfLiteModel(tflite::TensorType tensorType,
                                          std::vector<int32_t>& paramsShape,
                                          std::vector<int32_t>& indicesShape,
                                          const std::vector<int32_t>& expectedOutputShape,
                                          int32_t axis,
                                          float quantScale = 1.0f,
                                          int quantOffset = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));

    auto quantizationParameters =
             CreateQuantizationParameters(flatBufferBuilder,
                                          0,
                                          0,
                                          flatBufferBuilder.CreateVector<float>({quantScale}),
                                          flatBufferBuilder.CreateVector<int64_t>({quantOffset}));

    std::array<flatbuffers::Offset<Tensor>, 3> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(paramsShape.data(),
                                                                      paramsShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("params"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(indicesShape.data(),
                                                                      indicesShape.size()),
                              ::tflite::TensorType_INT32,
                              0,
                              flatBufferBuilder.CreateString("indices"),
                              quantizationParameters);
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(expectedOutputShape.data(),
                                                                      expectedOutputShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);


    // create operator
    tflite::BuiltinOptions    operatorBuiltinOptionsType = tflite::BuiltinOptions_GatherOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions     = CreateGatherOptions(flatBufferBuilder).Union();

    const std::vector<int>        operatorInputs{{0, 1}};
    const std::vector<int>        operatorOutputs{2};
    flatbuffers::Offset<Operator> controlOperator        =
                                      CreateOperator(flatBufferBuilder,
                                                     0,
                                                     flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(),
                                                                                             operatorInputs.size()),
                                                     flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(),
                                                                                             operatorOutputs.size()),
                                                     operatorBuiltinOptionsType,
                                                     operatorBuiltinOptions);

    const std::vector<int>        subgraphInputs{{0, 1}};
    const std::vector<int>        subgraphOutputs{2};
    flatbuffers::Offset<SubGraph> subgraph               =
                                      CreateSubGraph(flatBufferBuilder,
                                                     flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                                                     flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(),
                                                                                             subgraphInputs.size()),
                                                     flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(),
                                                                                             subgraphOutputs.size()),
                                                     flatBufferBuilder.CreateVector(&controlOperator, 1));

    flatbuffers::Offset<flatbuffers::String> modelDescription =
                                                 flatBufferBuilder.CreateString("ArmnnDelegate: GATHER Operator Model");
    flatbuffers::Offset<OperatorCode>        operatorCode     = CreateOperatorCode(flatBufferBuilder,
                                                                                   BuiltinOperator_GATHER);

    flatbuffers::Offset<Model> flatbufferModel =
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

template<typename T>
void GatherTest(tflite::TensorType tensorType,
                std::vector<armnn::BackendId>& backends,
                std::vector<int32_t>& paramsShape,
                std::vector<int32_t>& indicesShape,
                std::vector<int32_t>& expectedOutputShape,
                int32_t axis,
                std::vector<T>& paramsValues,
                std::vector<int32_t>& indicesValues,
                std::vector<T>& expectedOutputValues,
                float quantScale = 1.0f,
                int quantOffset = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateGatherTfLiteModel(tensorType,
                                                            paramsShape,
                                                            indicesShape,
                                                            expectedOutputShape,
                                                            axis,
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
    armnnDelegate::FillInput<T>(tfLiteDelegate, 0, paramsValues);
    armnnDelegate::FillInput<T>(armnnDelegate, 0, paramsValues);
    armnnDelegate::FillInput<int32_t>(tfLiteDelegate, 1, indicesValues);
    armnnDelegate::FillInput<int32_t>(armnnDelegate, 1, indicesValues);

    // Run EnqueWorkload
    CHECK(tfLiteDelegate->Invoke() == kTfLiteOk);
    CHECK(armnnDelegate->Invoke() == kTfLiteOk);

    // Compare output data
    armnnDelegate::CompareOutputData<T>(tfLiteDelegate,
                                        armnnDelegate,
                                        expectedOutputShape,
                                        expectedOutputValues,
                                        0);

    tfLiteDelegate.reset(nullptr);
    armnnDelegate.reset(nullptr);
}
} // anonymous namespace