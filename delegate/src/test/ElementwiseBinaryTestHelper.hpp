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
std::vector<char> CreateElementwiseBinaryTfLiteModel(tflite::BuiltinOperator binaryOperatorCode,
                                                     tflite::ActivationFunctionType activationType,
                                                     tflite::TensorType tensorType,
                                                     const std::vector <int32_t>& input0TensorShape,
                                                     const std::vector <int32_t>& input1TensorShape,
                                                     const std::vector <int32_t>& outputTensorShape,
                                                     std::vector<T>& input1Values,
                                                     bool constantInput = false,
                                                     float quantScale = 1.0f,
                                                     int quantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));
    if (constantInput)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(input1Values.data()),
                                                        sizeof(T) * input1Values.size())));
    }
    else
    {
        buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));
    }
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
                              1,
                              flatBufferBuilder.CreateString("input_1"),
                              quantizationParameters);
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              2,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_NONE;
    flatbuffers::Offset<void> operatorBuiltinOptions = 0;
    switch (binaryOperatorCode)
    {
        case BuiltinOperator_ADD:
        {
            operatorBuiltinOptionsType = BuiltinOptions_AddOptions;
            operatorBuiltinOptions = CreateAddOptions(flatBufferBuilder, activationType).Union();
            break;
        }
        case BuiltinOperator_DIV:
        {
            operatorBuiltinOptionsType = BuiltinOptions_DivOptions;
            operatorBuiltinOptions = CreateDivOptions(flatBufferBuilder, activationType).Union();
            break;
        }
        case BuiltinOperator_MAXIMUM:
        {
            operatorBuiltinOptionsType = BuiltinOptions_MaximumMinimumOptions;
            operatorBuiltinOptions = CreateMaximumMinimumOptions(flatBufferBuilder).Union();
            break;
        }
        case BuiltinOperator_MINIMUM:
        {
            operatorBuiltinOptionsType = BuiltinOptions_MaximumMinimumOptions;
            operatorBuiltinOptions = CreateMaximumMinimumOptions(flatBufferBuilder).Union();
            break;
        }
        case BuiltinOperator_MUL:
        {
            operatorBuiltinOptionsType = BuiltinOptions_MulOptions;
            operatorBuiltinOptions = CreateMulOptions(flatBufferBuilder, activationType).Union();
            break;
        }
        case BuiltinOperator_SUB:
        {
            operatorBuiltinOptionsType = BuiltinOptions_SubOptions;
            operatorBuiltinOptions = CreateSubOptions(flatBufferBuilder, activationType).Union();
            break;
        }
        case BuiltinOperator_FLOOR_DIV:
        {
            operatorBuiltinOptionsType = tflite::BuiltinOptions_FloorDivOptions;
            operatorBuiltinOptions = CreateSubOptions(flatBufferBuilder, activationType).Union();
            break;
        }
        default:
            break;
    }
    const std::vector<int32_t> operatorInputs{0, 1};
    const std::vector<int32_t> operatorOutputs{2};
    flatbuffers::Offset <Operator> elementwiseBinaryOperator =
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
                       flatBufferBuilder.CreateVector(&elementwiseBinaryOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Elementwise Binary Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, binaryOperatorCode);

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
void ElementwiseBinaryTest(tflite::BuiltinOperator binaryOperatorCode,
                           tflite::ActivationFunctionType activationType,
                           tflite::TensorType tensorType,
                           std::vector<armnn::BackendId>& backends,
                           std::vector<int32_t>& input0Shape,
                           std::vector<int32_t>& input1Shape,
                           std::vector<int32_t>& outputShape,
                           std::vector<T>& input0Values,
                           std::vector<T>& input1Values,
                           std::vector<T>& expectedOutputValues,
                           float quantScale = 1.0f,
                           int quantOffset  = 0,
                           bool constantInput = false)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateElementwiseBinaryTfLiteModel<T>(binaryOperatorCode,
                                                                          activationType,
                                                                          tensorType,
                                                                          input0Shape,
                                                                          input1Shape,
                                                                          outputShape,
                                                                          input1Values,
                                                                          constantInput,
                                                                          quantScale,
                                                                          quantOffset);

    const Model* tfLiteModel = GetModel(modelBuffer.data());
    // Create TfLite Interpreters
    std::unique_ptr <Interpreter> armnnDelegateInterpreter;
    CHECK(InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
              (&armnnDelegateInterpreter) == kTfLiteOk);
    CHECK(armnnDelegateInterpreter != nullptr);
    CHECK(armnnDelegateInterpreter->AllocateTensors() == kTfLiteOk);

    std::unique_ptr <Interpreter> tfLiteInterpreter;
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
    if (!constantInput)
    {
        armnnDelegate::FillInput<T>(tfLiteInterpreter, 1, input1Values);
        armnnDelegate::FillInput<T>(armnnDelegateInterpreter, 1, input1Values);
    }
    // Run EnqueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    // Compare output data
    armnnDelegate::CompareOutputData<T>(tfLiteInterpreter,
                                        armnnDelegateInterpreter,
                                        outputShape,
                                        expectedOutputValues);
    armnnDelegateInterpreter.reset(nullptr);
}

} // anonymous namespace