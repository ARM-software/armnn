//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn_delegate.hpp>
#include <armnnUtils/FloatingPointComparison.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

namespace
{
std::vector<char> CreateSoftmaxTfLiteModel(tflite::BuiltinOperator softmaxOperatorCode,
                                           tflite::TensorType tensorType,
                                           const std::vector <int32_t>& tensorShape,
                                           float beta)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));

    std::array<flatbuffers::Offset<Tensor>, 2> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              0);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              0);

    const std::vector<int32_t> operatorInputs({0});
    const std::vector<int32_t> operatorOutputs({1});

    flatbuffers::Offset<Operator> softmaxOperator;
    flatbuffers::Offset<flatbuffers::String> modelDescription;
    flatbuffers::Offset<OperatorCode> operatorCode;

    switch (softmaxOperatorCode)
    {
        case tflite::BuiltinOperator_SOFTMAX:
            softmaxOperator =
                CreateOperator(flatBufferBuilder,
                               0,
                               flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                               flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                               BuiltinOptions_SoftmaxOptions,
                               CreateSoftmaxOptions(flatBufferBuilder, beta).Union());
                modelDescription = flatBufferBuilder.CreateString("ArmnnDelegate: Softmax Operator Model");
                operatorCode = CreateOperatorCode(flatBufferBuilder,
                                 tflite::BuiltinOperator_SOFTMAX);
            break;
        case tflite::BuiltinOperator_LOG_SOFTMAX:
            softmaxOperator =
                CreateOperator(flatBufferBuilder,
                               0,
                               flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                               flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                               BuiltinOptions_LogSoftmaxOptions,
                               CreateLogSoftmaxOptions(flatBufferBuilder).Union());
                flatBufferBuilder.CreateString("ArmnnDelegate: Log-Softmax Operator Model");
            operatorCode = CreateOperatorCode(flatBufferBuilder,
                                              tflite::BuiltinOperator_LOG_SOFTMAX);
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
                       flatBufferBuilder.CreateVector(&softmaxOperator, 1));
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

void SoftmaxTest(tflite::BuiltinOperator softmaxOperatorCode,
                 tflite::TensorType tensorType,
                 std::vector<armnn::BackendId>& backends,
                 std::vector<int32_t>& shape,
                 std::vector<float>& inputValues,
                 std::vector<float>& expectedOutputValues,
                 float beta = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateSoftmaxTfLiteModel(softmaxOperatorCode,
                                                             tensorType,
                                                             shape,
                                                             beta);

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
    auto tfLiteInterpreterInputData = tfLiteInterpreter->typed_tensor<float>(tfLiteDelegateInputId);
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        tfLiteInterpreterInputData[i] = inputValues[i];
    }

    auto armnnDelegateInputId = armnnDelegateInterpreter->inputs()[0];
    auto armnnDelegateInputData = armnnDelegateInterpreter->typed_tensor<float>(armnnDelegateInputId);
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        armnnDelegateInputData[i] = inputValues[i];
    }
    // Run EnqueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    // Compare output data
    auto tfLiteInterpreterOutputId = tfLiteInterpreter->outputs()[0];
    auto tfLiteInterpreterOutputData = tfLiteInterpreter->typed_tensor<float>(tfLiteInterpreterOutputId);
    auto armnnDelegateOutputId = armnnDelegateInterpreter->outputs()[0];
    auto armnnDelegateOutputData = armnnDelegateInterpreter->typed_tensor<float>(armnnDelegateOutputId);

    for (size_t i = 0; i < inputValues.size(); ++i)
    {
         CHECK(armnnUtils::within_percentage_tolerance(expectedOutputValues[i], armnnDelegateOutputData[i], 0.1));
         CHECK(armnnUtils::within_percentage_tolerance(tfLiteInterpreterOutputData[i],
                                                       armnnDelegateOutputData[i], 0.1));
    }
}


/// Convenience function to run softmax and log-softmax test cases
/// \param operatorCode tflite::BuiltinOperator_SOFTMAX or tflite::BuiltinOperator_LOG_SOFTMAX
/// \param backends armnn backends to target
/// \param beta multiplicative parameter to the softmax function
/// \param expectedOutput to be checked against transformed input
void SoftmaxTestCase(tflite::BuiltinOperator operatorCode,
                     std::vector<armnn::BackendId> backends, float beta, std::vector<float> expectedOutput) {
    std::vector<float> input = {
        1.0, 2.5, 3.0, 4.5, 5.0,
        -1.0, -2.5, -3.0, -4.5, -5.0};
    std::vector<int32_t> shape = {2, 5};

    SoftmaxTest(operatorCode,
                tflite::TensorType_FLOAT32,
                backends,
                shape,
                input,
                expectedOutput,
                beta);
}

} // anonymous namespace
