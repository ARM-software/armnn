//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn_delegate.hpp>
#include <DelegateTestInterpreter.hpp>

#include <tensorflow/lite/version.h>

namespace
{

std::vector<char> CreateScatterNdTfLiteModel(tflite::TensorType tensorType,
                                             const std::vector<int32_t>& indicesShape,
                                             const std::vector<int32_t>& updatesShape,
                                             const std::vector<int32_t>& shapeShape,
                                             const std::vector<int32_t>& outputShape,
                                             const std::vector<int32_t>& shapeData,
                                             float quantScale = 1.0f,
                                             int quantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder)); // indices
    buffers.push_back(CreateBuffer(flatBufferBuilder)); // updates
    buffers.push_back(CreateBuffer(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(shapeData.data()),
                                                                      sizeof(int32_t) * shapeData.size())));
    buffers.push_back(CreateBuffer(flatBufferBuilder)); // output

    auto quantizationParameters =
            CreateQuantizationParameters(flatBufferBuilder,
                                         0,
                                         0,
                                         flatBufferBuilder.CreateVector<float>({ quantScale }),
                                         flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));

    std::array<flatbuffers::Offset<Tensor>, 4> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(indicesShape.data(),
                                                                      indicesShape.size()),
                              TensorType_INT32,
                              1,
                              flatBufferBuilder.CreateString("indices_tensor"),
                              quantizationParameters);

    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(updatesShape.data(),
                                                                      updatesShape.size()),
                              tensorType,
                              2,
                              flatBufferBuilder.CreateString("updates_tensor"),
                              quantizationParameters);

    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(shapeShape.data(),
                                                                      shapeShape.size()),
                              TensorType_INT32,
                              3,
                              flatBufferBuilder.CreateString("shape_tensor"),
                              quantizationParameters);

    tensors[3] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputShape.data(),
                                                                      outputShape.size()),
                              tensorType,
                              4,
                              flatBufferBuilder.CreateString("output_tensor"),
                              quantizationParameters);

    // Create Operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_ScatterNdOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions  = CreateScatterNdOptions(flatBufferBuilder).Union();

    const std::vector<int> operatorInputs { 0, 1, 2 };
    const std::vector<int> operatorOutputs { 3 };

    flatbuffers::Offset<Operator> scatterNdOperator =
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
                           flatBufferBuilder.CreateVector(&scatterNdOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: ScatterNd Operator Model");
    flatbuffers::Offset <OperatorCode> opCode = CreateOperatorCode(flatBufferBuilder,
                                                                   tflite::BuiltinOperator_SCATTER_ND);

    flatbuffers::Offset <Model> flatbufferModel =
            CreateModel(flatBufferBuilder,
                        TFLITE_SCHEMA_VERSION,
                        flatBufferBuilder.CreateVector(&opCode, 1),
                        flatBufferBuilder.CreateVector(&subgraph, 1),
                        modelDescription,
                        flatBufferBuilder.CreateVector(buffers.data(), buffers.size()));

    flatBufferBuilder.Finish(flatbufferModel, armnnDelegate::FILE_IDENTIFIER);

    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

template<typename T>
void ScatterNdTestImpl(tflite::TensorType tensorType,
                       std::vector<int32_t>& indicesShape,
                       std::vector<int32_t>& indicesValues,
                       std::vector<int32_t>& updatesShape,
                       std::vector<T>& updatesValues,
                       std::vector<int32_t>& shapeShape,
                       std::vector<int32_t>& shapeValue,
                       std::vector<int32_t>& expectedOutputShape,
                       std::vector<T>& expectedOutputValues,
                       const std::vector<armnn::BackendId>& backends = {},
                       float quantScale = 1.0f,
                       int quantOffset = 0)
{
    using namespace delegateTestInterpreter;

    std::vector<char> modelBuffer = CreateScatterNdTfLiteModel(tensorType,
                                                               indicesShape,
                                                               updatesShape,
                                                               shapeShape,
                                                               expectedOutputShape,
                                                               shapeValue,
                                                               quantScale,
                                                               quantOffset);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<int32_t>(indicesValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(updatesValues, 1) == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<int32_t>(shapeValue, 2) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>   tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, CaptureAvailableBackends(backends));
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<int32_t>(indicesValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(updatesValues, 1) == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<int32_t>(shapeValue, 2) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>   armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, expectedOutputShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

} // anonymous namespace