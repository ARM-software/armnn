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
std::vector<char> CreatePadTfLiteModel(
    tflite::BuiltinOperator padOperatorCode,
    tflite::TensorType tensorType,
    tflite::MirrorPadMode paddingMode,
    const std::vector<int32_t>& inputTensorShape,
    const std::vector<int32_t>& paddingTensorShape,
    const std::vector<int32_t>& outputTensorShape,
    const std::vector<int32_t>& paddingDim,
    const std::vector<T> paddingValue,
    float quantScale = 1.0f,
    int quantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

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
                                    0,
                                    flatBufferBuilder.CreateString("input"),
                                    quantizationParameters);

    auto paddingTensor = CreateTensor(flatBufferBuilder,
                                      flatBufferBuilder.CreateVector<int32_t>(paddingTensorShape.data(),
                                                                              paddingTensorShape.size()),
                                      tflite::TensorType_INT32,
                                      1,
                                      flatBufferBuilder.CreateString("padding"));

    auto outputTensor = CreateTensor(flatBufferBuilder,
                                     flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                             outputTensorShape.size()),
                                     tensorType,
                                     2,
                                     flatBufferBuilder.CreateString("output"),
                                     quantizationParameters);

    std::vector<flatbuffers::Offset<Tensor>> tensors = { inputTensor, paddingTensor, outputTensor};

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(paddingDim.data()),
                                                    sizeof(int32_t) * paddingDim.size())));
    buffers.push_back(CreateBuffer(flatBufferBuilder));

    std::vector<int32_t> operatorInputs;
    std::vector<int> subgraphInputs;

    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_PadOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions;

    if (padOperatorCode == tflite::BuiltinOperator_PAD)
    {
        operatorInputs = {{ 0, 1 }};
        subgraphInputs = {{ 0, 1 }};
        operatorBuiltinOptions = CreatePadOptions(flatBufferBuilder).Union();
    }
    else if(padOperatorCode == tflite::BuiltinOperator_MIRROR_PAD)
    {
        operatorInputs = {{ 0, 1 }};
        subgraphInputs = {{ 0, 1 }};

        operatorBuiltinOptionsType = BuiltinOptions_MirrorPadOptions;
        operatorBuiltinOptions = CreateMirrorPadOptions(flatBufferBuilder, paddingMode).Union();
    }
    else if (padOperatorCode == tflite::BuiltinOperator_PADV2)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(paddingValue.data()),
                                                        sizeof(T))));

        const std::vector<int32_t> shape = { 1 };
        auto padValueTensor = CreateTensor(flatBufferBuilder,
                                           flatBufferBuilder.CreateVector<int32_t>(shape.data(),
                                                                                   shape.size()),
                                           tensorType,
                                           3,
                                           flatBufferBuilder.CreateString("paddingValue"),
                                           quantizationParameters);

        tensors.push_back(padValueTensor);

        operatorInputs = {{ 0, 1, 3 }};
        subgraphInputs = {{ 0, 1, 3 }};

        operatorBuiltinOptionsType = BuiltinOptions_PadV2Options;
        operatorBuiltinOptions = CreatePadV2Options(flatBufferBuilder).Union();
    }

    // create operator
    const std::vector<int32_t> operatorOutputs{ 2 };
    flatbuffers::Offset <Operator> paddingOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       operatorBuiltinOptionsType,
                       operatorBuiltinOptions);

    const std::vector<int> subgraphOutputs{ 2 };
    flatbuffers::Offset <SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&paddingOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Pad Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         padOperatorCode);

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
void PadTest(tflite::BuiltinOperator padOperatorCode,
             tflite::TensorType tensorType,
             const std::vector<armnn::BackendId>& backends,
             const std::vector<int32_t>& inputShape,
             const std::vector<int32_t>& paddingShape,
             std::vector<int32_t>& outputShape,
             std::vector<T>& inputValues,
             std::vector<int32_t>& paddingDim,
             std::vector<T>& expectedOutputValues,
             T paddingValue,
             float quantScale = 1.0f,
             int quantOffset  = 0,
             tflite::MirrorPadMode paddingMode = tflite::MirrorPadMode_SYMMETRIC)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreatePadTfLiteModel<T>(padOperatorCode,
                                                            tensorType,
                                                            paddingMode,
                                                            inputShape,
                                                            paddingShape,
                                                            outputShape,
                                                            paddingDim,
                                                            {paddingValue},
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
