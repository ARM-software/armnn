//
// Copyright © 2020, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn_delegate.hpp>
#include <DelegateTestInterpreter.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/version.h>

#include <schema_generated.h>

#include <doctest/doctest.h>

namespace
{

std::vector<char> CreateElementwiseUnaryTfLiteModel(tflite::BuiltinOperator unaryOperatorCode,
                                                    tflite::TensorType tensorType,
                                                    const std::vector <int32_t>& tensorShape)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::array<flatbuffers::Offset<tflite::Buffer>, 1> buffers;
    buffers[0] = CreateBuffer(flatBufferBuilder);

    std::array<flatbuffers::Offset<Tensor>, 2> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(), tensorShape.size()),
                              tensorType);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(), tensorShape.size()),
                              tensorType);

    // create operator
    const std::vector<int> operatorInputs{0};
    const std::vector<int> operatorOutputs{1};
    flatbuffers::Offset <Operator> unaryOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()));

    const std::vector<int> subgraphInputs{0};
    const std::vector<int> subgraphOutputs{1};
    flatbuffers::Offset <SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&unaryOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Elementwise Unary Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, unaryOperatorCode);

    flatbuffers::Offset <Model> flatbufferModel =
        CreateModel(flatBufferBuilder,
                    TFLITE_SCHEMA_VERSION,
                    flatBufferBuilder.CreateVector(&operatorCode, 1),
                    flatBufferBuilder.CreateVector(&subgraph, 1),
                    modelDescription,
                    flatBufferBuilder.CreateVector(buffers.data(), buffers.size()));

    flatBufferBuilder.Finish(flatbufferModel, armnnDelegate::FILE_IDENTIFIER);

    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

void ElementwiseUnaryFP32Test(tflite::BuiltinOperator unaryOperatorCode,
                              std::vector<armnn::BackendId>& backends,
                              std::vector<float>& inputValues,
                              std::vector<float>& expectedOutputValues)
{
    using namespace delegateTestInterpreter;
    std::vector<int32_t> inputShape  { { 3, 1, 2} };
    std::vector<char> modelBuffer = CreateElementwiseUnaryTfLiteModel(unaryOperatorCode,
                                                                      ::tflite::TensorType_FLOAT32,
                                                                      inputShape);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<float>(inputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<float>   tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<float>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<float>(inputValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<float>   armnnOutputValues = armnnInterpreter.GetOutputResult<float>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<float>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, inputShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

void ElementwiseUnaryBoolTest(tflite::BuiltinOperator unaryOperatorCode,
                              std::vector<armnn::BackendId>& backends,
                              std::vector<int32_t>& inputShape,
                              std::vector<bool>& inputValues,
                              std::vector<bool>& expectedOutputValues)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateElementwiseUnaryTfLiteModel(unaryOperatorCode,
                                                                      ::tflite::TensorType_BOOL,
                                                                      inputShape);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor(inputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<bool>    tfLiteOutputValues = tfLiteInterpreter.GetOutputResult(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor(inputValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<bool>    armnnOutputValues = armnnInterpreter.GetOutputResult(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareData(expectedOutputValues, armnnOutputValues, expectedOutputValues.size());
    armnnDelegate::CompareData(expectedOutputValues, tfLiteOutputValues, expectedOutputValues.size());
    armnnDelegate::CompareData(tfLiteOutputValues, armnnOutputValues, expectedOutputValues.size());

    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, inputShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

} // anonymous namespace




