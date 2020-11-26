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

std::vector<char> CreateLogicalBinaryTfLiteModel(tflite::BuiltinOperator logicalOperatorCode,
                                                 tflite::TensorType tensorType,
                                                 const std::vector <int32_t>& input0TensorShape,
                                                 const std::vector <int32_t>& input1TensorShape,
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
                                     flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));


    std::array<flatbuffers::Offset<Tensor>, 3> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(input0TensorShape.data(),
                                                                      input0TensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("input_0"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(input1TensorShape.data(),
                                                                      input1TensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("input_1"),
                              quantizationParameters);
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_NONE;
    flatbuffers::Offset<void> operatorBuiltinOptions = 0;
    switch (logicalOperatorCode)
    {
        case BuiltinOperator_LOGICAL_AND:
        {
            operatorBuiltinOptionsType = BuiltinOptions_LogicalAndOptions;
            operatorBuiltinOptions = CreateLogicalAndOptions(flatBufferBuilder).Union();
            break;
        }
        case BuiltinOperator_LOGICAL_OR:
        {
            operatorBuiltinOptionsType = BuiltinOptions_LogicalOrOptions;
            operatorBuiltinOptions = CreateLogicalOrOptions(flatBufferBuilder).Union();
            break;
        }
        default:
            break;
    }
    const std::vector<int32_t> operatorInputs{ {0, 1} };
    const std::vector<int32_t> operatorOutputs{ 2 };
    flatbuffers::Offset <Operator> logicalBinaryOperator =
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
                       flatBufferBuilder.CreateVector(&logicalBinaryOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Logical Binary Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, logicalOperatorCode);

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
void LogicalBinaryTest(tflite::BuiltinOperator logicalOperatorCode,
                       tflite::TensorType tensorType,
                       std::vector<armnn::BackendId>& backends,
                       std::vector<int32_t>& input0Shape,
                       std::vector<int32_t>& input1Shape,
                       std::vector<int32_t>& expectedOutputShape,
                       std::vector<T>& input0Values,
                       std::vector<T>& input1Values,
                       std::vector<T>& expectedOutputValues,
                       float quantScale = 1.0f,
                       int quantOffset  = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateLogicalBinaryTfLiteModel(logicalOperatorCode,
                                                                   tensorType,
                                                                   input0Shape,
                                                                   input1Shape,
                                                                   expectedOutputShape,
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

    // Set input data for the armnn interpreter
    armnnDelegate::FillInput(armnnDelegateInterpreter, 0, input0Values);
    armnnDelegate::FillInput(armnnDelegateInterpreter, 1, input1Values);

    // Set input data for the tflite interpreter
    armnnDelegate::FillInput(tfLiteInterpreter, 0, input0Values);
    armnnDelegate::FillInput(tfLiteInterpreter, 1, input1Values);

    // Run EnqueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    // Compare output data, comparing Boolean values is handled differently and needs to call the CompareData function
    // directly. This is because Boolean types get converted to a bit representation in a vector.
    auto tfLiteDelegateOutputId = tfLiteInterpreter->outputs()[0];
    auto tfLiteDelegateOutputData = tfLiteInterpreter->typed_tensor<T>(tfLiteDelegateOutputId);
    auto armnnDelegateOutputId = armnnDelegateInterpreter->outputs()[0];
    auto armnnDelegateOutputData = armnnDelegateInterpreter->typed_tensor<T>(armnnDelegateOutputId);

    armnnDelegate::CompareData(expectedOutputValues, armnnDelegateOutputData, expectedOutputValues.size());
    armnnDelegate::CompareData(expectedOutputValues, tfLiteDelegateOutputData, expectedOutputValues.size());
    armnnDelegate::CompareData(tfLiteDelegateOutputData, armnnDelegateOutputData, expectedOutputValues.size());

    armnnDelegateInterpreter.reset(nullptr);
    tfLiteInterpreter.reset(nullptr);
}

} // anonymous namespace