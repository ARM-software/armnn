//
// Copyright © 2020, 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn_delegate.hpp>
#include <DelegateTestInterpreter.hpp>

#include <tensorflow/lite/version.h>

namespace
{

std::vector<char> CreateSplitTfLiteModel(tflite::TensorType tensorType,
                                         std::vector<int32_t>& axisTensorShape,
                                         std::vector<int32_t>& inputTensorShape,
                                         const std::vector<std::vector<int32_t>>& outputTensorShapes,
                                         std::vector<int32_t>& axisData,
                                         const int32_t numSplits,
                                         float quantScale = 1.0f,
                                         int quantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(axisData.data()),
                                                                  sizeof(int32_t) * axisData.size())));

    auto quantizationParameters =
            CreateQuantizationParameters(flatBufferBuilder,
                                         0,
                                         0,
                                         flatBufferBuilder.CreateVector<float>({ quantScale }),
                                         flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));

    std::array<flatbuffers::Offset<Tensor>, 4> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(axisTensorShape.data(),
                                                                      axisTensorShape.size()),
                              ::tflite::TensorType_INT32,
                              2,
                              flatBufferBuilder.CreateString("axis"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              tensorType,
                              1,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);

    // Create output tensor
    for (unsigned int i = 0; i < outputTensorShapes.size(); ++i)
    {
        buffers.push_back(CreateBuffer(flatBufferBuilder));
        tensors[i + 2] = CreateTensor(flatBufferBuilder,
                                      flatBufferBuilder.CreateVector<int32_t>(outputTensorShapes[i].data(),
                                                                              outputTensorShapes[i].size()),
                                      tensorType,
                                      (i+3),
                                      flatBufferBuilder.CreateString("output"),
                                      quantizationParameters);
    }

    // create operator. Mean uses ReducerOptions.
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_SplitOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateSplitOptions(flatBufferBuilder, numSplits).Union();

    const std::vector<int> operatorInputs{ {0, 1} };
    const std::vector<int> operatorOutputs{ {2, 3} };
    flatbuffers::Offset <Operator> controlOperator =
            CreateOperator(flatBufferBuilder,
                           0,
                           flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                           operatorBuiltinOptionsType,
                           operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{ {0, 1} };
    const std::vector<int> subgraphOutputs{ {2, 3} };
    flatbuffers::Offset <SubGraph> subgraph =
            CreateSubGraph(flatBufferBuilder,
                           flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                           flatBufferBuilder.CreateVector(&controlOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: SPLIT Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, BuiltinOperator_SPLIT);

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
void SplitTest(tflite::TensorType tensorType,
               std::vector<int32_t>& axisTensorShape,
               std::vector<int32_t>& inputTensorShape,
               std::vector<std::vector<int32_t>>& outputTensorShapes,
               std::vector<int32_t>& axisData,
               std::vector<T>& inputValues,
               std::vector<std::vector<T>>& expectedOutputValues,
               const int32_t numSplits,
               const std::vector<armnn::BackendId>& backends = {},
               float quantScale = 1.0f,
               int quantOffset  = 0)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateSplitTfLiteModel(tensorType,
                                                           axisTensorShape,
                                                           inputTensorShape,
                                                           outputTensorShapes,
                                                           axisData,
                                                           numSplits,
                                                           quantScale,
                                                           quantOffset);
    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(inputValues, 1) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, CaptureAvailableBackends(backends));
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(inputValues, 1) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);

    // Compare output data
    for (unsigned int i = 0; i < expectedOutputValues.size(); ++i)
    {
        std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(i);
        std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(i);

        std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(i);
        std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(i);

        armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues[i]);
        armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, outputTensorShapes[i]);
    }

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();

} // End of SPLIT Test

std::vector<char> CreateSplitVTfLiteModel(tflite::TensorType tensorType,
                                          std::vector<int32_t>& inputTensorShape,
                                          std::vector<int32_t>& splitsTensorShape,
                                          std::vector<int32_t>& axisTensorShape,
                                          const std::vector<std::vector<int32_t>>& outputTensorShapes,
                                          std::vector<int32_t>& splitsData,
                                          std::vector<int32_t>& axisData,
                                          const int32_t numSplits,
                                          float quantScale = 1.0f,
                                          int quantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::array<flatbuffers::Offset<tflite::Buffer>, 3> buffers;
    buffers[0] = CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({}));
    buffers[1] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(splitsData.data()),
                                                             sizeof(int32_t) * splitsData.size()));
    buffers[2] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(axisData.data()),
                                                             sizeof(int32_t) * axisData.size()));

    auto quantizationParameters =
            CreateQuantizationParameters(flatBufferBuilder,
                                         0,
                                         0,
                                         flatBufferBuilder.CreateVector<float>({ quantScale }),
                                         flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));

    std::array<flatbuffers::Offset<Tensor>, 5> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(splitsTensorShape.data(),
                                                                      splitsTensorShape.size()),
                              ::tflite::TensorType_INT32,
                              1,
                              flatBufferBuilder.CreateString("splits"),
                              quantizationParameters);
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(axisTensorShape.data(),
                                                                      axisTensorShape.size()),
                              ::tflite::TensorType_INT32,
                              2,
                              flatBufferBuilder.CreateString("axis"),
                              quantizationParameters);

    // Create output tensor
    for (unsigned int i = 0; i < outputTensorShapes.size(); ++i)
    {
        tensors[i + 3] = CreateTensor(flatBufferBuilder,
                                      flatBufferBuilder.CreateVector<int32_t>(outputTensorShapes[i].data(),
                                                                              outputTensorShapes[i].size()),
                                      tensorType,
                                      0,
                                      flatBufferBuilder.CreateString("output"),
                                      quantizationParameters);
    }

    // create operator. Mean uses ReducerOptions.
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_SplitVOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateSplitVOptions(flatBufferBuilder, numSplits).Union();

    const std::vector<int> operatorInputs{ {0, 1, 2} };
    std::vector<int> operatorOutputs;

    for (uint32_t i = 0; i< outputTensorShapes.size(); ++i)
    {
        operatorOutputs.emplace_back(i+3);
    }


    flatbuffers::Offset <Operator> controlOperator =
            CreateOperator(flatBufferBuilder,
                           0,
                           flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                           operatorBuiltinOptionsType,
                           operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{ {0, 1, 2} };
    std::vector<int> subgraphOutputs;

    for (uint32_t i = 0; i< outputTensorShapes.size(); ++i)
    {
        subgraphOutputs.emplace_back(i+3);
    }

    flatbuffers::Offset <SubGraph> subgraph =
            CreateSubGraph(flatBufferBuilder,
                           flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                           flatBufferBuilder.CreateVector(&controlOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: SPLIT_V Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, BuiltinOperator_SPLIT_V);

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
void SplitVTest(tflite::TensorType tensorType,
                std::vector<int32_t>& inputTensorShape,
                std::vector<int32_t>& splitsTensorShape,
                std::vector<int32_t>& axisTensorShape,
                std::vector<std::vector<int32_t>>& outputTensorShapes,
                std::vector<T>& inputValues,
                std::vector<int32_t>& splitsData,
                std::vector<int32_t>& axisData,
                std::vector<std::vector<T>>& expectedOutputValues,
                const int32_t numSplits,
                const std::vector<armnn::BackendId>& backends = {},
                float quantScale = 1.0f,
                int quantOffset  = 0)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateSplitVTfLiteModel(tensorType,
                                                            inputTensorShape,
                                                            splitsTensorShape,
                                                            axisTensorShape,
                                                            outputTensorShapes,
                                                            splitsData,
                                                            axisData,
                                                            numSplits,
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
        armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, outputTensorShapes[i]);
    }

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
} // End of SPLIT_V Test

} // anonymous namespace