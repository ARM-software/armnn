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
std::vector<char> CreateSpaceDepthTfLiteModel(tflite::BuiltinOperator spaceDepthOperatorCode,
                                              tflite::TensorType tensorType,
                                              const std::vector <int32_t>& inputTensorShape,
                                              const std::vector <int32_t>& outputTensorShape,
                                              int32_t blockSize)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    auto quantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({  1.0f }),
                                     flatBufferBuilder.CreateVector<int64_t>({ 0 }));

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));

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

    const std::vector<int32_t> operatorInputs({0});
    const std::vector<int32_t> operatorOutputs({1});

    flatbuffers::Offset<Operator> spaceDepthOperator;
    flatbuffers::Offset<flatbuffers::String> modelDescription;
    flatbuffers::Offset<OperatorCode> operatorCode;

    switch (spaceDepthOperatorCode)
    {
        case tflite::BuiltinOperator_SPACE_TO_DEPTH:
            spaceDepthOperator =
                CreateOperator(flatBufferBuilder,
                               0,
                               flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                               flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                               BuiltinOptions_SpaceToDepthOptions,
                               CreateSpaceToDepthOptions(flatBufferBuilder, blockSize).Union());
                modelDescription = flatBufferBuilder.CreateString("ArmnnDelegate: SPACE_TO_DEPTH Operator Model");
                operatorCode = CreateOperatorCode(flatBufferBuilder,
                                 tflite::BuiltinOperator_SPACE_TO_DEPTH);
            break;
        case tflite::BuiltinOperator_DEPTH_TO_SPACE:
            spaceDepthOperator =
                CreateOperator(flatBufferBuilder,
                               0,
                               flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                               flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                               BuiltinOptions_DepthToSpaceOptions,
                               CreateDepthToSpaceOptions(flatBufferBuilder, blockSize).Union());
                flatBufferBuilder.CreateString("ArmnnDelegate: DEPTH_TO_SPACE Operator Model");
            operatorCode = CreateOperatorCode(flatBufferBuilder,
                                              tflite::BuiltinOperator_DEPTH_TO_SPACE);
            break;
        default:
            break;
    }
    const std::vector<int32_t> subgraphInputs({0});
    const std::vector<int32_t> subgraphOutputs({1});
    flatbuffers::Offset<SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&spaceDepthOperator, 1));
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

template <typename T>
void SpaceDepthTest(tflite::BuiltinOperator spaceDepthOperatorCode,
                    tflite::TensorType tensorType,
                    std::vector<armnn::BackendId>& backends,
                    std::vector<int32_t>& inputShape,
                    std::vector<int32_t>& outputShape,
                    std::vector<T>& inputValues,
                    std::vector<T>& expectedOutputValues,
                    int32_t blockSize = 2)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateSpaceDepthTfLiteModel(spaceDepthOperatorCode,
                                                                tensorType,
                                                                inputShape,
                                                                outputShape,
                                                                blockSize);

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

    // Run EnqueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    // Compare output data
    armnnDelegate::CompareOutputData(tfLiteInterpreter, armnnDelegateInterpreter, outputShape, expectedOutputValues);
}

} // anonymous namespace
