//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn_delegate.hpp>
#include <armnn/DescriptorsFwd.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

#include <string>

namespace
{

std::vector<char> CreateSliceTfLiteModel(tflite::TensorType tensorType,
                                         const std::vector<int32_t>& inputTensorShape,
                                         const std::vector<int32_t>& beginTensorData,
                                         const std::vector<int32_t>& sizeTensorData,
                                         const std::vector<int32_t>& beginTensorShape,
                                         const std::vector<int32_t>& sizeTensorShape,
                                         const std::vector<int32_t>& outputTensorShape)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    flatbuffers::Offset<tflite::Buffer> buffers[5] = {
            CreateBuffer(flatBufferBuilder),
            CreateBuffer(flatBufferBuilder),
            CreateBuffer(flatBufferBuilder,
            flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(beginTensorData.data()),
            sizeof(int32_t) * beginTensorData.size())),
            CreateBuffer(flatBufferBuilder,
            flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(sizeTensorData.data()),
            sizeof(int32_t) * sizeTensorData.size())),
            CreateBuffer(flatBufferBuilder)
    };

    std::array<flatbuffers::Offset<Tensor>, 4> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              tensorType,
                              1,
                              flatBufferBuilder.CreateString("input"));
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(beginTensorShape.data(),
                                                                      beginTensorShape.size()),
                              ::tflite::TensorType_INT32,
                              2,
                              flatBufferBuilder.CreateString("begin_tensor"));
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(sizeTensorShape.data(),
                                                                      sizeTensorShape.size()),
                              ::tflite::TensorType_INT32,
                              3,
                              flatBufferBuilder.CreateString("size_tensor"));
    tensors[3] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              4,
                              flatBufferBuilder.CreateString("output"));


    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_SliceOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateSliceOptions(flatBufferBuilder).Union();

    const std::vector<int> operatorInputs{ 0, 1, 2 };
    const std::vector<int> operatorOutputs{ 3 };
    flatbuffers::Offset <Operator> sliceOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       operatorBuiltinOptionsType,
                       operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{ 0, 1, 2 };
    const std::vector<int> subgraphOutputs{ 3 };
    flatbuffers::Offset <SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&sliceOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Slice Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         BuiltinOperator_SLICE);

    flatbuffers::Offset <Model> flatbufferModel =
        CreateModel(flatBufferBuilder,
                    TFLITE_SCHEMA_VERSION,
                    flatBufferBuilder.CreateVector(&operatorCode, 1),
                    flatBufferBuilder.CreateVector(&subgraph, 1),
                    modelDescription,
                    flatBufferBuilder.CreateVector(buffers, 5));

    flatBufferBuilder.Finish(flatbufferModel);

    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

template <typename T>
void SliceTestImpl(std::vector<armnn::BackendId>& backends,
                   std::vector<T>& inputValues,
                   std::vector<T>& expectedOutputValues,
                   std::vector<int32_t>& beginTensorData,
                   std::vector<int32_t>& sizeTensorData,
                   std::vector<int32_t>& inputTensorShape,
                   std::vector<int32_t>& beginTensorShape,
                   std::vector<int32_t>& sizeTensorShape,
                   std::vector<int32_t>& outputTensorShape)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateSliceTfLiteModel(
        ::tflite::TensorType_FLOAT32,
        inputTensorShape,
        beginTensorData,
        sizeTensorData,
        beginTensorShape,
        sizeTensorShape,
        outputTensorShape);

    auto tfLiteModel = GetModel(modelBuffer.data());

    // Create TfLite Interpreters
    std::unique_ptr<Interpreter> armnnDelegate;
    CHECK(InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
        (&armnnDelegate) == kTfLiteOk);
    CHECK(armnnDelegate != nullptr);
    CHECK(armnnDelegate->AllocateTensors() == kTfLiteOk);

    std::unique_ptr<Interpreter> tfLiteDelegate;
    CHECK(InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
        (&tfLiteDelegate) == kTfLiteOk);
    CHECK(tfLiteDelegate != nullptr);
    CHECK(tfLiteDelegate->AllocateTensors() == kTfLiteOk);

    // Create the ArmNN Delegate
    armnnDelegate::DelegateOptions delegateOptions(backends);
    std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
        theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                         armnnDelegate::TfLiteArmnnDelegateDelete);
    CHECK(theArmnnDelegate != nullptr);

    // Modify armnnDelegateInterpreter to use armnnDelegate
    CHECK(armnnDelegate->ModifyGraphWithDelegate(theArmnnDelegate.get()) == kTfLiteOk);

    // Set input data
    armnnDelegate::FillInput<T>(tfLiteDelegate, 0, inputValues);
    armnnDelegate::FillInput<T>(armnnDelegate, 0, inputValues);

    // Run EnqueWorkload
    CHECK(tfLiteDelegate->Invoke() == kTfLiteOk);
    CHECK(armnnDelegate->Invoke() == kTfLiteOk);

    // Compare output data
    armnnDelegate::CompareOutputData<T>(tfLiteDelegate,
                                        armnnDelegate,
                                        outputTensorShape,
                                        expectedOutputValues);

    tfLiteDelegate.reset(nullptr);
    armnnDelegate.reset(nullptr);
} // End of Slice Test

} // anonymous namespace