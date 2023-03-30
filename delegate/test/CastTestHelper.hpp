//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
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
std::vector<char> CreateCastTfLiteModel(tflite::TensorType inputTensorType,
                                        tflite::TensorType outputTensorType,
                                        const std::vector <int32_t>& tensorShape,
                                        float quantScale = 1.0f,
                                        int quantOffset = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));

    auto quantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({quantScale}),
                                     flatBufferBuilder.CreateVector<int64_t>({quantOffset}));

    std::array<flatbuffers::Offset<Tensor>, 2> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              inputTensorType,
                              1,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              outputTensorType,
                              2,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    const std::vector<int32_t> operatorInputs({0});
    const std::vector<int32_t> operatorOutputs({1});

    flatbuffers::Offset<Operator> castOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       BuiltinOptions_CastOptions,
                       CreateCastOptions(flatBufferBuilder).Union());

    flatbuffers::Offset<flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: CAST Operator Model");
    flatbuffers::Offset<OperatorCode> operatorCode =
        CreateOperatorCode(flatBufferBuilder, tflite::BuiltinOperator_CAST);

    const std::vector<int32_t> subgraphInputs({0});
    const std::vector<int32_t> subgraphOutputs({1});
    flatbuffers::Offset<SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&castOperator, 1));

    flatbuffers::Offset<Model> flatbufferModel =
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

template<typename T, typename K>
void CastTest(tflite::TensorType inputTensorType,
              tflite::TensorType outputTensorType,
              std::vector<armnn::BackendId>& backends,
              std::vector<int32_t>& shape,
              std::vector<T>& inputValues,
              std::vector<K>& expectedOutputValues,
              float quantScale = 1.0f,
              int quantOffset = 0)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateCastTfLiteModel(inputTensorType,
                                                          outputTensorType,
                                                          shape,
                                                          quantScale,
                                                          quantOffset);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<K>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<K>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<K>       armnnOutputValues = armnnInterpreter.GetOutputResult<K>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<K>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, shape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

} // anonymous namespace
