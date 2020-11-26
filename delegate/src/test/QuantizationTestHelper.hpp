//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

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

std::vector<char> CreateQuantizationTfLiteModel(tflite::BuiltinOperator quantizationOperatorCode,
                                                tflite::TensorType inputTensorType,
                                                tflite::TensorType outputTensorType,
                                                const std::vector <int32_t>& inputTensorShape,
                                                const std::vector <int32_t>& outputTensorShape,
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
                                         flatBufferBuilder.CreateVector<int64_t>({ quantOffset }),
                                         QuantizationDetails_CustomQuantization);

    std::array<flatbuffers::Offset<Tensor>, 2> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              inputTensorType,
                              0,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              outputTensorType,
                              0,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_NONE;
    flatbuffers::Offset<void> operatorBuiltinOptions = 0;
    switch (quantizationOperatorCode)
    {
        case BuiltinOperator_QUANTIZE:
        {
            operatorBuiltinOptionsType = BuiltinOptions_QuantizeOptions;
            operatorBuiltinOptions = CreateQuantizeOptions(flatBufferBuilder).Union();
            break;
        }
        case BuiltinOperator_DEQUANTIZE:
        {
            operatorBuiltinOptionsType = BuiltinOptions_DequantizeOptions;
            operatorBuiltinOptions = CreateDequantizeOptions(flatBufferBuilder).Union();
            break;
        }
        default:
            break;
    }

    const std::vector<int32_t> operatorInputs{0};
    const std::vector<int32_t> operatorOutputs{1};
    flatbuffers::Offset <Operator> quantizationOperator =
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
                           flatBufferBuilder.CreateVector(&quantizationOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: Quantization Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, quantizationOperatorCode);

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
void QuantizationTest(tflite::BuiltinOperator quantizeOperatorCode,
                      tflite::TensorType inputTensorType,
                      tflite::TensorType outputTensorType,
                      std::vector<armnn::BackendId>& backends,
                      std::vector<int32_t>& inputShape,
                      std::vector<int32_t>& outputShape,
                      std::vector<InputT>&  inputValues,
                      std::vector<OutputT>& expectedOutputValues,
                      float quantScale = 1.0f,
                      int quantOffset  = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateQuantizationTfLiteModel(quantizeOperatorCode,
                                                                  inputTensorType,
                                                                  outputTensorType,
                                                                  inputShape,
                                                                  outputShape,
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
    auto tfLiteDelegateInputId = tfLiteInterpreter->inputs()[0];
    auto tfLiteDelageInputData = tfLiteInterpreter->typed_tensor<InputT>(tfLiteDelegateInputId);
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        tfLiteDelageInputData[i] = inputValues[i];
    }

    auto armnnDelegateInputId = armnnDelegateInterpreter->inputs()[0];
    auto armnnDelegateInputData = armnnDelegateInterpreter->typed_tensor<InputT>(armnnDelegateInputId);
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        armnnDelegateInputData[i] = inputValues[i];
    }

    // Run EnqueWorkload
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