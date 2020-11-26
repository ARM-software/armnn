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

std::vector<char> CreateComparisonTfLiteModel(tflite::BuiltinOperator comparisonOperatorCode,
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
                              ::tflite::TensorType_BOOL,
                              0);

    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_EqualOptions;;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateEqualOptions(flatBufferBuilder).Union();
    switch (comparisonOperatorCode)
    {
        case BuiltinOperator_EQUAL:
        {
            operatorBuiltinOptionsType = BuiltinOptions_EqualOptions;
            operatorBuiltinOptions = CreateEqualOptions(flatBufferBuilder).Union();
            break;
        }
        case BuiltinOperator_NOT_EQUAL:
        {
            operatorBuiltinOptionsType = BuiltinOptions_NotEqualOptions;
            operatorBuiltinOptions = CreateNotEqualOptions(flatBufferBuilder).Union();
            break;
        }
        case BuiltinOperator_GREATER:
        {
            operatorBuiltinOptionsType = BuiltinOptions_GreaterOptions;
            operatorBuiltinOptions = CreateGreaterOptions(flatBufferBuilder).Union();
            break;
        }
        case BuiltinOperator_GREATER_EQUAL:
        {
            operatorBuiltinOptionsType = BuiltinOptions_GreaterEqualOptions;
            operatorBuiltinOptions = CreateGreaterEqualOptions(flatBufferBuilder).Union();
            break;
        }
        case BuiltinOperator_LESS:
        {
            operatorBuiltinOptionsType = BuiltinOptions_LessOptions;
            operatorBuiltinOptions = CreateLessOptions(flatBufferBuilder).Union();
            break;
        }
        case BuiltinOperator_LESS_EQUAL:
        {
            operatorBuiltinOptionsType = BuiltinOptions_LessEqualOptions;
            operatorBuiltinOptions = CreateLessEqualOptions(flatBufferBuilder).Union();
            break;
        }
        default:
            break;
    }
    const std::vector<int32_t> operatorInputs{0, 1};
    const std::vector<int32_t> operatorOutputs{2};
    flatbuffers::Offset <Operator> comparisonOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       operatorBuiltinOptionsType,
                       operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{0, 1};
    const std::vector<int> subgraphOutputs{2};
    flatbuffers::Offset <SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&comparisonOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Comparison Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, comparisonOperatorCode);

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
void ComparisonTest(tflite::BuiltinOperator comparisonOperatorCode,
                    tflite::TensorType tensorType,
                    std::vector<armnn::BackendId>& backends,
                    std::vector<int32_t>& input0Shape,
                    std::vector<int32_t>& input1Shape,
                    std::vector<int32_t>& outputShape,
                    std::vector<T>& input0Values,
                    std::vector<T>& input1Values,
                    std::vector<bool>& expectedOutputValues,
                    float quantScale = 1.0f,
                    int quantOffset  = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateComparisonTfLiteModel(comparisonOperatorCode,
                                                                tensorType,
                                                                input0Shape,
                                                                input1Shape,
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
    auto tfLiteDelegateInput0Id = tfLiteInterpreter->inputs()[0];
    auto tfLiteDelageInput0Data = tfLiteInterpreter->typed_tensor<T>(tfLiteDelegateInput0Id);
    for (unsigned int i = 0; i < input0Values.size(); ++i)
    {
        tfLiteDelageInput0Data[i] = input0Values[i];
    }

    auto tfLiteDelegateInput1Id = tfLiteInterpreter->inputs()[1];
    auto tfLiteDelageInput1Data = tfLiteInterpreter->typed_tensor<T>(tfLiteDelegateInput1Id);
    for (unsigned int i = 0; i < input1Values.size(); ++i)
    {
        tfLiteDelageInput1Data[i] = input1Values[i];
    }

    auto armnnDelegateInput0Id = armnnDelegateInterpreter->inputs()[0];
    auto armnnDelegateInput0Data = armnnDelegateInterpreter->typed_tensor<T>(armnnDelegateInput0Id);
    for (unsigned int i = 0; i < input0Values.size(); ++i)
    {
        armnnDelegateInput0Data[i] = input0Values[i];
    }

    auto armnnDelegateInput1Id = armnnDelegateInterpreter->inputs()[1];
    auto armnnDelegateInput1Data = armnnDelegateInterpreter->typed_tensor<T>(armnnDelegateInput1Id);
    for (unsigned int i = 0; i < input1Values.size(); ++i)
    {
        armnnDelegateInput1Data[i] = input1Values[i];
    }

    // Run EnqueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);
    // Compare output data
    auto tfLiteDelegateOutputId = tfLiteInterpreter->outputs()[0];
    auto tfLiteDelageOutputData = tfLiteInterpreter->typed_tensor<bool>(tfLiteDelegateOutputId);
    auto armnnDelegateOutputId = armnnDelegateInterpreter->outputs()[0];
    auto armnnDelegateOutputData = armnnDelegateInterpreter->typed_tensor<bool>(armnnDelegateOutputId);

    armnnDelegate::CompareData(expectedOutputValues  , armnnDelegateOutputData, expectedOutputValues.size());
    armnnDelegate::CompareData(expectedOutputValues  , tfLiteDelageOutputData , expectedOutputValues.size());
    armnnDelegate::CompareData(tfLiteDelageOutputData, armnnDelegateOutputData, expectedOutputValues.size());
}

} // anonymous namespace