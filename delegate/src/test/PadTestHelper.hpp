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

template <typename T>
std::vector<char> CreatePadTfLiteModel(
    tflite::BuiltinOperator padOperatorCode,
    tflite::TensorType tensorType,
    tflite::MirrorPadMode paddingMode,
    const std::vector<int32_t>& inputTensorShape,
    const std::vector<int32_t>& paddingTensorShape,
    const std::vector<int32_t>& outputTensorShape,
    const std::vector<int32_t>& paddingDim,
    const std::vector<T> paddingValue,
    float quantScale = 1.0f,
    int quantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

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

    auto paddingTensor = CreateTensor(flatBufferBuilder,
                                      flatBufferBuilder.CreateVector<int32_t>(paddingTensorShape.data(),
                                                                              paddingTensorShape.size()),
                                      tflite::TensorType_INT32,
                                      1,
                                      flatBufferBuilder.CreateString("padding"));

    auto outputTensor = CreateTensor(flatBufferBuilder,
                                     flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                             outputTensorShape.size()),
                                     tensorType,
                                     2,
                                     flatBufferBuilder.CreateString("output"),
                                     quantizationParameters);

    std::vector<flatbuffers::Offset<Tensor>> tensors = { inputTensor, paddingTensor, outputTensor};

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));
    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(paddingDim.data()),
                                                    sizeof(int32_t) * paddingDim.size())));
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));

    std::vector<int32_t> operatorInputs;
    std::vector<int> subgraphInputs;

    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_PadOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions;

    if (padOperatorCode == tflite::BuiltinOperator_PAD)
    {
        operatorInputs = {{ 0, 1 }};
        subgraphInputs = {{ 0, 1 }};
        operatorBuiltinOptions = CreatePadOptions(flatBufferBuilder).Union();
    }
    else if(padOperatorCode == tflite::BuiltinOperator_MIRROR_PAD)
    {
        operatorInputs = {{ 0, 1 }};
        subgraphInputs = {{ 0, 1 }};

        operatorBuiltinOptionsType = BuiltinOptions_MirrorPadOptions;
        operatorBuiltinOptions = CreateMirrorPadOptions(flatBufferBuilder, paddingMode).Union();
    }
    else if (padOperatorCode == tflite::BuiltinOperator_PADV2)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(paddingValue.data()),
                                                        sizeof(T))));

        const std::vector<int32_t> shape = { 1 };
        auto padValueTensor = CreateTensor(flatBufferBuilder,
                                           flatBufferBuilder.CreateVector<int32_t>(shape.data(),
                                                                                   shape.size()),
                                           tensorType,
                                           3,
                                           flatBufferBuilder.CreateString("paddingValue"),
                                           quantizationParameters);

        tensors.push_back(padValueTensor);

        operatorInputs = {{ 0, 1, 3 }};
        subgraphInputs = {{ 0, 1, 3 }};

        operatorBuiltinOptionsType = BuiltinOptions_PadV2Options;
        operatorBuiltinOptions = CreatePadV2Options(flatBufferBuilder).Union();
    }

    // create operator
    const std::vector<int32_t> operatorOutputs{ 2 };
    flatbuffers::Offset <Operator> paddingOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       operatorBuiltinOptionsType,
                       operatorBuiltinOptions);

    const std::vector<int> subgraphOutputs{ 2 };
    flatbuffers::Offset <SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&paddingOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Pad Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         padOperatorCode);

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
void PadTest(tflite::BuiltinOperator padOperatorCode,
             tflite::TensorType tensorType,
             const std::vector<armnn::BackendId>& backends,
             const std::vector<int32_t>& inputShape,
             const std::vector<int32_t>& paddingShape,
             std::vector<int32_t>& outputShape,
             std::vector<T>& inputValues,
             std::vector<int32_t>& paddingDim,
             std::vector<T>& expectedOutputValues,
             T paddingValue,
             float quantScale = 1.0f,
             int quantOffset  = 0,
             tflite::MirrorPadMode paddingMode = tflite::MirrorPadMode_SYMMETRIC)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreatePadTfLiteModel<T>(padOperatorCode,
                                                            tensorType,
                                                            paddingMode,
                                                            inputShape,
                                                            paddingShape,
                                                            outputShape,
                                                            paddingDim,
                                                            {paddingValue},
                                                            quantScale,
                                                            quantOffset);

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
    armnnDelegate::FillInput<T>(tfLiteInterpreter, 0, inputValues);
    armnnDelegate::FillInput<T>(armnnDelegateInterpreter, 0, inputValues);

    // Run EnqueueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    armnnDelegate::CompareOutputData<T>(tfLiteInterpreter, armnnDelegateInterpreter, outputShape, expectedOutputValues);
}

} // anonymous namespace
