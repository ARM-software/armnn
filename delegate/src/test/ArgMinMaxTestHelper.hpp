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

template <typename InputT, typename OutputT>
std::vector<char> CreateArgMinMaxTfLiteModel(tflite::BuiltinOperator argMinMaxOperatorCode,
                                             tflite::TensorType tensorType,
                                             const std::vector<int32_t>& inputTensorShape,
                                             const std::vector<int32_t>& axisTensorShape,
                                             const std::vector<int32_t>& outputTensorShape,
                                             const std::vector<OutputT> axisValue,
                                             tflite::TensorType outputType,
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

    auto axisTensor = CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(axisTensorShape.data(),
                                                                           axisTensorShape.size()),
                                   tflite::TensorType_INT32,
                                   1,
                                   flatBufferBuilder.CreateString("axis"));

    auto outputTensor = CreateTensor(flatBufferBuilder,
                                     flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                             outputTensorShape.size()),
                                     outputType,
                                     2,
                                     flatBufferBuilder.CreateString("output"),
                                     quantizationParameters);

    std::vector<flatbuffers::Offset<Tensor>> tensors = { inputTensor, axisTensor, outputTensor };

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));
    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(axisValue.data()),
                                                    sizeof(OutputT))));
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));

    std::vector<int32_t> operatorInputs = {{ 0, 1 }};
    std::vector<int> subgraphInputs = {{ 0, 1 }};

    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_ArgMaxOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateArgMaxOptions(flatBufferBuilder, outputType).Union();

    if (argMinMaxOperatorCode == tflite::BuiltinOperator_ARG_MIN)
    {
        operatorBuiltinOptionsType = BuiltinOptions_ArgMinOptions;
        operatorBuiltinOptions = CreateArgMinOptions(flatBufferBuilder, outputType).Union();
    }

    // create operator
    const std::vector<int32_t> operatorOutputs{ 2 };
    flatbuffers::Offset <Operator> argMinMaxOperator =
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
                       flatBufferBuilder.CreateVector(&argMinMaxOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: ArgMinMax Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         argMinMaxOperatorCode);

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

template <typename InputT, typename OutputT>
void ArgMinMaxTest(tflite::BuiltinOperator argMinMaxOperatorCode,
                   tflite::TensorType tensorType,
                   const std::vector<armnn::BackendId>& backends,
                   const std::vector<int32_t>& inputShape,
                   const std::vector<int32_t>& axisShape,
                   std::vector<int32_t>& outputShape,
                   std::vector<InputT>& inputValues,
                   std::vector<OutputT>& expectedOutputValues,
                   OutputT axisValue,
                   tflite::TensorType outputType,
                   float quantScale = 1.0f,
                   int quantOffset  = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateArgMinMaxTfLiteModel<InputT, OutputT>(argMinMaxOperatorCode,
                                                                                tensorType,
                                                                                inputShape,
                                                                                axisShape,
                                                                                outputShape,
                                                                                {axisValue},
                                                                                outputType,
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
    armnnDelegate::FillInput<InputT>(tfLiteInterpreter, 0, inputValues);
    armnnDelegate::FillInput<InputT>(armnnDelegateInterpreter, 0, inputValues);

    // Run EnqueueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    // Compare output data
    auto tfLiteDelegateOutputId = tfLiteInterpreter->outputs()[0];
    auto tfLiteDelageOutputData = tfLiteInterpreter->typed_tensor<OutputT>(tfLiteDelegateOutputId);
    auto armnnDelegateOutputId = armnnDelegateInterpreter->outputs()[0];
    auto armnnDelegateOutputData = armnnDelegateInterpreter->typed_tensor<OutputT>(armnnDelegateOutputId);

    for (size_t i = 0; i < expectedOutputValues.size(); i++)
    {
        CHECK(expectedOutputValues[i] == armnnDelegateOutputData[i]);
        CHECK(tfLiteDelageOutputData[i] == expectedOutputValues[i]);
        CHECK(tfLiteDelageOutputData[i] == armnnDelegateOutputData[i]);
    }
}

} // anonymous namespace