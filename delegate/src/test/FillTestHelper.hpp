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

template <typename T>
std::vector<char> CreateFillTfLiteModel(tflite::BuiltinOperator fillOperatorCode,
                                        tflite::TensorType tensorType,
                                        const std::vector<int32_t>& inputShape,
                                        const std::vector <int32_t>& tensorShape,
                                        const std::vector<T> fillValue)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector({})));
    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(tensorShape.data()),
                                                    sizeof(int32_t) * tensorShape.size())));
    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(fillValue.data()),
                                                    sizeof(T) * fillValue.size())));

    std::array<flatbuffers::Offset<Tensor>, 3> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputShape.data(),
                                                                      inputShape.size()),
                              tflite::TensorType_INT32,
                              1,
                              flatBufferBuilder.CreateString("dims"));

    std::vector<int32_t> fillShape = {};
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(fillShape.data(),
                                                                      fillShape.size()),
                              tensorType,
                              2,
                              flatBufferBuilder.CreateString("value"));

    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("output"));

    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_FillOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateFillOptions(flatBufferBuilder).Union();

    // create operator
    const std::vector<int> operatorInputs{ {0, 1} };
    const std::vector<int> operatorOutputs{ 2 };
    flatbuffers::Offset <Operator> fillOperator =
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
                       flatBufferBuilder.CreateVector(&fillOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Fill Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         fillOperatorCode);

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
void FillTest(tflite::BuiltinOperator fillOperatorCode,
              tflite::TensorType tensorType,
              const std::vector<armnn::BackendId>& backends,
              std::vector<int32_t >& inputShape,
              std::vector<int32_t >& tensorShape,
              std::vector<T>& expectedOutputValues,
              T fillValue)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateFillTfLiteModel<T>(fillOperatorCode,
                                                             tensorType,
                                                             inputShape,
                                                             tensorShape,
                                                             {fillValue});

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

    // Run EnqueueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    armnnDelegate::CompareOutputData<T>(tfLiteInterpreter, armnnDelegateInterpreter, tensorShape, expectedOutputValues);
}

} // anonymous namespace
