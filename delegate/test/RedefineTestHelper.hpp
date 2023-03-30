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

std::vector<char> CreateRedefineTfLiteModel(
        tflite::BuiltinOperator redefineOperatorCode,
        tflite::TensorType tensorType,
        const std::vector<int32_t>& inputTensorShape,
        const std::vector<int32_t>& outputTensorShape,
        const std::vector<int32_t>& targetShape,
        bool useOption = true,
        float quantScale = 1.0f,
        int quantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));

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
                                    1,
                                    flatBufferBuilder.CreateString("input"),
                                    quantizationParameters);

    std::vector<flatbuffers::Offset<Tensor>> tensors;
    std::vector<int32_t> operatorInputs;
    std::vector<int> subgraphInputs;
    flatbuffers::Offset<void> operatorBuiltinOptions;

    if (useOption)
    {
        buffers.push_back(CreateBuffer(flatBufferBuilder));
        auto outputTensor = CreateTensor(flatBufferBuilder,
                                         flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                                 outputTensorShape.size()),
                                         tensorType,
                                         2,
                                         flatBufferBuilder.CreateString("output"),
                                         quantizationParameters);
        tensors = { inputTensor, outputTensor};
        operatorInputs = {0};
        subgraphInputs = {0};
        operatorBuiltinOptions = CreateReshapeOptions(
                flatBufferBuilder,
                flatBufferBuilder.CreateVector(targetShape.data(), targetShape.size())).Union();
    }
    else
    {
        buffers.push_back(
                CreateBuffer(flatBufferBuilder,
                             flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(targetShape.data()),
                                                            sizeof(int32_t) * targetShape.size())));
        int32_t size = static_cast<int32_t>(targetShape.size());
        auto shapeTensor = CreateTensor(flatBufferBuilder,
                                        flatBufferBuilder.CreateVector<int32_t>( { size } ),
                                        tflite::TensorType_INT32,
                                        2,
                                        flatBufferBuilder.CreateString("shape"));

        buffers.push_back(CreateBuffer(flatBufferBuilder));
        auto outputTensor = CreateTensor(flatBufferBuilder,
                                         flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                                 outputTensorShape.size()),
                                         tensorType,
                                         3,
                                         flatBufferBuilder.CreateString("output"),
                                         quantizationParameters);

        tensors = { inputTensor, outputTensor, shapeTensor };
        operatorInputs = {0, 2};
        subgraphInputs = {0, 2};
        operatorBuiltinOptions = CreateReshapeOptions(flatBufferBuilder).Union();
    }

    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_ReshapeOptions;

    const std::vector<int32_t> operatorOutputs{1};
    flatbuffers::Offset <Operator> redefineOperator =
            CreateOperator(flatBufferBuilder,
                           0,
                           flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                           operatorBuiltinOptionsType,
                           operatorBuiltinOptions);

    const std::vector<int> subgraphOutputs{1};
    flatbuffers::Offset <SubGraph> subgraph =
            CreateSubGraph(flatBufferBuilder,
                           flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                           flatBufferBuilder.CreateVector(&redefineOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: Reshape Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         redefineOperatorCode);

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

template <typename T>
void RedefineTest(tflite::BuiltinOperator redefineOperatorCode,
                  tflite::TensorType tensorType,
                  const std::vector<armnn::BackendId>& backends,
                  const std::vector<int32_t>& inputShape,
                  std::vector<int32_t>& outputShape,
                  std::vector<T>& inputValues,
                  std::vector<T>& expectedOutputValues,
                  std::vector<int32_t>& targetShape,
                  bool useOption = true,
                  float quantScale = 1.0f,
                  int quantOffset  = 0)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateRedefineTfLiteModel(redefineOperatorCode,
                                                              tensorType,
                                                              inputShape,
                                                              outputShape,
                                                              targetShape,
                                                              useOption,
                                                              quantScale,
                                                              quantOffset);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, outputShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

} // anonymous namespace