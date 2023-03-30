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

template <typename T>
std::vector<char> CreateFullyConnectedTfLiteModel(tflite::TensorType tensorType,
                                                  tflite::ActivationFunctionType activationType,
                                                  const std::vector <int32_t>& inputTensorShape,
                                                  const std::vector <int32_t>& weightsTensorShape,
                                                  const std::vector <int32_t>& biasTensorShape,
                                                  std::vector <int32_t>& outputTensorShape,
                                                  std::vector <T>& weightsData,
                                                  bool constantWeights = true,
                                                  float quantScale = 1.0f,
                                                  int quantOffset  = 0,
                                                  float outputQuantScale = 2.0f,
                                                  int outputQuantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;
    std::array<flatbuffers::Offset<tflite::Buffer>, 5> buffers;
    buffers[0] = CreateBuffer(flatBufferBuilder);
    buffers[1] = CreateBuffer(flatBufferBuilder);

    auto biasTensorType = ::tflite::TensorType_FLOAT32;
    if (tensorType == ::tflite::TensorType_INT8)
    {
        biasTensorType = ::tflite::TensorType_INT32;
    }
    if (constantWeights)
    {
        buffers[2] = CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(weightsData.data()),
                                                    sizeof(T) * weightsData.size()));

        if (tensorType == ::tflite::TensorType_INT8)
        {
            std::vector<int32_t> biasData = { 10 };
            buffers[3] = CreateBuffer(flatBufferBuilder,
                                      flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(biasData.data()),
                                                                     sizeof(int32_t) * biasData.size()));

        }
        else
        {
            std::vector<float> biasData = { 10 };
            buffers[3] = CreateBuffer(flatBufferBuilder,
                                      flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(biasData.data()),
                                                                     sizeof(float) * biasData.size()));
        }
    }
    else
    {
        buffers[2] = CreateBuffer(flatBufferBuilder);
        buffers[3] = CreateBuffer(flatBufferBuilder);
    }
    buffers[4] = CreateBuffer(flatBufferBuilder);

    auto quantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({ quantScale }),
                                     flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));

    auto outputQuantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({ outputQuantScale }),
                                     flatBufferBuilder.CreateVector<int64_t>({ outputQuantOffset }));

    std::array<flatbuffers::Offset<Tensor>, 4> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              tensorType,
                              1,
                              flatBufferBuilder.CreateString("input_0"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(weightsTensorShape.data(),
                                                                      weightsTensorShape.size()),
                              tensorType,
                              2,
                              flatBufferBuilder.CreateString("weights"),
                              quantizationParameters);
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(biasTensorShape.data(),
                                                                      biasTensorShape.size()),
                              biasTensorType,
                              3,
                              flatBufferBuilder.CreateString("bias"),
                              quantizationParameters);

    tensors[3] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              4,
                              flatBufferBuilder.CreateString("output"),
                              outputQuantizationParameters);


    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_FullyConnectedOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions =
        CreateFullyConnectedOptions(flatBufferBuilder,
                                    activationType,
                                    FullyConnectedOptionsWeightsFormat_DEFAULT, false).Union();

    const std::vector<int> operatorInputs{0, 1, 2};
    const std::vector<int> operatorOutputs{3};
    flatbuffers::Offset <Operator> fullyConnectedOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       operatorBuiltinOptionsType, operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{0, 1, 2};
    const std::vector<int> subgraphOutputs{3};
    flatbuffers::Offset <SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&fullyConnectedOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: FullyConnected Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         tflite::BuiltinOperator_FULLY_CONNECTED);

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
void FullyConnectedTest(std::vector<armnn::BackendId>& backends,
                        tflite::TensorType tensorType,
                        tflite::ActivationFunctionType activationType,
                        const std::vector <int32_t>& inputTensorShape,
                        const std::vector <int32_t>& weightsTensorShape,
                        const std::vector <int32_t>& biasTensorShape,
                        std::vector <int32_t>& outputTensorShape,
                        std::vector <T>& inputValues,
                        std::vector <T>& expectedOutputValues,
                        std::vector <T>& weightsData,
                        bool constantWeights = true,
                        float quantScale = 1.0f,
                        int quantOffset  = 0)
{
    using namespace delegateTestInterpreter;

    std::vector<char> modelBuffer = CreateFullyConnectedTfLiteModel(tensorType,
                                                                    activationType,
                                                                    inputTensorShape,
                                                                    weightsTensorShape,
                                                                    biasTensorShape,
                                                                    outputTensorShape,
                                                                    weightsData,
                                                                    constantWeights,
                                                                    quantScale,
                                                                    quantOffset);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);

    CHECK(tfLiteInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);

    if (!constantWeights)
    {
        CHECK(tfLiteInterpreter.FillInputTensor<T>(weightsData, 1) == kTfLiteOk);
        CHECK(armnnInterpreter.FillInputTensor<T>(weightsData, 1) == kTfLiteOk);

        if (tensorType == ::tflite::TensorType_INT8)
        {
            std::vector <int32_t> biasData = {10};
            CHECK(tfLiteInterpreter.FillInputTensor<int32_t>(biasData, 2) == kTfLiteOk);
            CHECK(armnnInterpreter.FillInputTensor<int32_t>(biasData, 2) == kTfLiteOk);
        }
        else
        {
            std::vector<float> biasData = {10};
            CHECK(tfLiteInterpreter.FillInputTensor<float>(biasData, 2) == kTfLiteOk);
            CHECK(armnnInterpreter.FillInputTensor<float>(biasData, 2) == kTfLiteOk);
        }
    }

    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, outputTensorShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

} // anonymous namespace