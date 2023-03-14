//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <schema_generated.h>
#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

namespace
{
std::vector<char> CreateBatchMatMulTfLiteModel(
        tflite::BuiltinOperator bmmOperatorCode,
        tflite::TensorType tensorType,
        const std::vector <int32_t>& LHSInputTensorShape,
        const std::vector <int32_t>& RHSInputTensorShape,
        const std::vector <int32_t>& outputTensorShape,
        bool adjX = false,
        bool adjY = false,
        float quantScale = 1.0f,
        int quantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));

    auto quantizationParameters =
            CreateQuantizationParameters(flatBufferBuilder,
                                         0,
                                         0,
                                         flatBufferBuilder.CreateVector<float>({ quantScale }),
                                         flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));

    std::array<flatbuffers::Offset<Tensor>, 3> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(LHSInputTensorShape.data(),
                                                                      LHSInputTensorShape.size()),
                              tensorType,
                              1,
                              flatBufferBuilder.CreateString("LHSInput"),
                              quantizationParameters);

    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(RHSInputTensorShape.data(),
                                                                      RHSInputTensorShape.size()),
                              tensorType,
                              2,
                              flatBufferBuilder.CreateString("RHSInput"),
                              quantizationParameters);

    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              3,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_BatchMatMulOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateBatchMatMulOptions(flatBufferBuilder,
                                                                                adjX,
                                                                                adjY).Union();

    const std::vector<int32_t> operatorInputs{{0, 1}};
    const std::vector<int32_t> operatorOutputs{2};
    flatbuffers::Offset <Operator> bmmOperator =
            CreateOperator(flatBufferBuilder,
                           0,
                           flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(),
                                                                   operatorOutputs.size()),
                           operatorBuiltinOptionsType,
                           operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{{0, 1}};
    const std::vector<int> subgraphOutputs{2};
    flatbuffers::Offset <SubGraph> subgraph =
            CreateSubGraph(flatBufferBuilder,
                           flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(),
                                                                   subgraphOutputs.size()),
                           flatBufferBuilder.CreateVector(&bmmOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: BatchMatMul Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, bmmOperatorCode);

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
void BatchMatMulTest(tflite::BuiltinOperator bmmOperatorCode,
                   tflite::TensorType tensorType,
                   std::vector<armnn::BackendId>& backends,
                   std::vector<int32_t>& LHSInputShape,
                   std::vector<int32_t>& RHSInputShape,
                   std::vector<int32_t>& outputShape,
                   std::vector<T>& LHSInputValues,
                   std::vector<T>& RHSInputValues,
                   std::vector<T>& expectedOutputValues,
                   bool adjX = false,
                   bool adjY = false,
                   float quantScale = 1.0f,
                   int quantOffset  = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateBatchMatMulTfLiteModel(bmmOperatorCode,
                                                                 tensorType,
                                                                 LHSInputShape,
                                                                 RHSInputShape,
                                                                 outputShape,
                                                                 adjX,
                                                                 adjY,
                                                                 quantScale,
                                                                 quantOffset);

    const Model* tfLiteModel = GetModel(modelBuffer.data());
    CHECK(tfLiteModel != nullptr);
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
    auto tfLiteDelegateLHSInputId = tfLiteInterpreter->inputs()[0];
    auto tfLiteDelegateLHSInputData = tfLiteInterpreter->typed_tensor<T>(tfLiteDelegateLHSInputId);
    auto tfLiteDelegateRHSInputId = tfLiteInterpreter->inputs()[1];
    auto tfLiteDelegateRHSInputData = tfLiteInterpreter->typed_tensor<T>(tfLiteDelegateRHSInputId);
    for (unsigned int i = 0; i < LHSInputValues.size(); ++i)
    {
        tfLiteDelegateLHSInputData[i] = LHSInputValues[i];
    }
    for (unsigned int i = 0; i < RHSInputValues.size(); ++i)
    {
        tfLiteDelegateRHSInputData[i] = RHSInputValues[i];
    }

    auto armnnDelegateLHSInputId = armnnDelegateInterpreter->inputs()[0];
    auto armnnDelegateLHSInputData = armnnDelegateInterpreter->typed_tensor<T>(armnnDelegateLHSInputId);
    auto armnnDelegateRHSInputId = armnnDelegateInterpreter->inputs()[1];
    auto armnnDelegateRHSInputData = armnnDelegateInterpreter->typed_tensor<T>(armnnDelegateRHSInputId);
    for (unsigned int i = 0; i < LHSInputValues.size(); ++i)
    {
        armnnDelegateLHSInputData[i] = LHSInputValues[i];
    }
    for (unsigned int i = 0; i < RHSInputValues.size(); ++i)
    {
        armnnDelegateRHSInputData[i] = RHSInputValues[i];
    }
    // Run EnqueueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    armnnDelegate::CompareOutputData(tfLiteInterpreter, armnnDelegateInterpreter,
                                     outputShape, expectedOutputValues);
}

} // anonymous namespace




