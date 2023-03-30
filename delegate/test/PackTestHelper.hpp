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

std::vector<char> CreatePackTfLiteModel(tflite::BuiltinOperator packOperatorCode,
                                        tflite::TensorType tensorType,
                                        std::vector<int32_t>& inputTensorShape,
                                        const std::vector <int32_t>& outputTensorShape,
                                        const int32_t inputTensorNum,
                                        unsigned int axis = 0,
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

    std::vector<int32_t> operatorInputs{};
    const std::vector<int32_t> operatorOutputs{inputTensorNum};
    std::vector<int> subgraphInputs{};
    const std::vector<int> subgraphOutputs{inputTensorNum};

    std::vector<flatbuffers::Offset<Tensor>> tensors(inputTensorNum + 1);
    for (int i = 0; i < inputTensorNum; ++i)
    {
        tensors[i] = CreateTensor(flatBufferBuilder,
                                  flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                          inputTensorShape.size()),
                                  tensorType,
                                  1,
                                  flatBufferBuilder.CreateString("input" + std::to_string(i)),
                                  quantizationParameters);

        // Add number of inputs to vector.
        operatorInputs.push_back(i);
        subgraphInputs.push_back(i);
    }

    // Create output tensor
    tensors[inputTensorNum] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_PackOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions =
            CreatePackOptions(flatBufferBuilder, inputTensorNum, axis).Union();

    flatbuffers::Offset <Operator> packOperator =
            CreateOperator(flatBufferBuilder,
                           0,
                           flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                           operatorBuiltinOptionsType,
                           operatorBuiltinOptions);

    flatbuffers::Offset <SubGraph> subgraph =
            CreateSubGraph(flatBufferBuilder,
                           flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                           flatBufferBuilder.CreateVector(&packOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: Pack Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, packOperatorCode);

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
void PackTest(tflite::BuiltinOperator packOperatorCode,
              tflite::TensorType tensorType,
              std::vector<armnn::BackendId>& backends,
              std::vector<int32_t>& inputShape,
              std::vector<int32_t>& expectedOutputShape,
              std::vector<std::vector<T>>& inputValues,
              std::vector<T>& expectedOutputValues,
              unsigned int axis = 0,
              float quantScale = 1.0f,
              int quantOffset  = 0)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreatePackTfLiteModel(packOperatorCode,
                                                          tensorType,
                                                          inputShape,
                                                          expectedOutputShape,
                                                          inputValues.size(),
                                                          axis,
                                                          quantScale,
                                                          quantOffset);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);

    // Set input data for all input tensors.
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        auto inputTensorValues = inputValues[i];
        CHECK(tfLiteInterpreter.FillInputTensor<T>(inputTensorValues, i) == kTfLiteOk);
        CHECK(armnnInterpreter.FillInputTensor<T>(inputTensorValues, i) == kTfLiteOk);
    }

    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, expectedOutputShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

} // anonymous namespace