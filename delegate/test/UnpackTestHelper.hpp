//
// Copyright © 2021, 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn_delegate.hpp>
#include <DelegateTestInterpreter.hpp>

#include <tensorflow/lite/version.h>

namespace
{

std::vector<char> CreateUnpackTfLiteModel(tflite::BuiltinOperator unpackOperatorCode,
                                          tflite::TensorType tensorType,
                                          std::vector<int32_t>& inputTensorShape,
                                          const std::vector <int32_t>& outputTensorShape,
                                          const int32_t outputTensorNum,
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

    const std::vector<int32_t> operatorInputs{ 0 };
    std::vector<int32_t> operatorOutputs{};
    const std::vector<int> subgraphInputs{ 0 };
    std::vector<int> subgraphOutputs{};

    std::vector<flatbuffers::Offset<Tensor>> tensors(outputTensorNum + 1);

    // Create input tensor
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                                                                      tensorType,
                                                                      1,
                                                                      flatBufferBuilder.CreateString("input"),
                                                                      quantizationParameters);

    for (int i = 0; i < outputTensorNum; ++i)
    {
        tensors[i + 1] = CreateTensor(flatBufferBuilder,
                                  flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                          outputTensorShape.size()),
                                  tensorType,
                                      (i + 2),
                                  flatBufferBuilder.CreateString("output" + std::to_string(i)),
                                  quantizationParameters);

        buffers.push_back(CreateBuffer(flatBufferBuilder));
        operatorOutputs.push_back(i + 1);
        subgraphOutputs.push_back(i + 1);
    }

    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_UnpackOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions =
        CreateUnpackOptions(flatBufferBuilder, outputTensorNum, axis).Union();

    flatbuffers::Offset <Operator> unpackOperator =
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
                       flatBufferBuilder.CreateVector(&unpackOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Unpack Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, unpackOperatorCode);

    flatbuffers::Offset <Model> flatbufferModel =
        CreateModel(flatBufferBuilder,
                    TFLITE_SCHEMA_VERSION,
                    flatBufferBuilder.CreateVector(&operatorCode, 1),
                    flatBufferBuilder.CreateVector(&subgraph, 1),
                    modelDescription,
                    flatBufferBuilder.CreateVector(buffers));

    flatBufferBuilder.Finish(flatbufferModel, armnnDelegate::FILE_IDENTIFIER);

    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

template <typename T>
void UnpackTest(tflite::BuiltinOperator unpackOperatorCode,
              tflite::TensorType tensorType,
              std::vector<int32_t>& inputShape,
              std::vector<int32_t>& expectedOutputShape,
              std::vector<T>& inputValues,
              std::vector<std::vector<T>>& expectedOutputValues,
              const std::vector<armnn::BackendId>& backends = {},
              unsigned int axis = 0,
              float quantScale = 1.0f,
              int quantOffset  = 0)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateUnpackTfLiteModel(unpackOperatorCode,
                                                            tensorType,
                                                            inputShape,
                                                            expectedOutputShape,
                                                            expectedOutputValues.size(),
                                                            axis,
                                                            quantScale,
                                                            quantOffset);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, CaptureAvailableBackends(backends));
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);

    // Compare output data
    for (unsigned int i = 0; i < expectedOutputValues.size(); ++i)
    {
        std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(i);
        std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(i);

        std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(i);
        std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(i);

        armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues[i]);
        armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, expectedOutputShape);
    }

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

} // anonymous namespace