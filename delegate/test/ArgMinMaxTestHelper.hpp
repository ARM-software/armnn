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

template <typename InputT, typename OutputT>
std::vector<char> CreateArgMinMaxTfLiteModel(tflite::BuiltinOperator argMinMaxOperatorCode,
                                             tflite::TensorType tensorType,
                                             const std::vector<int32_t>& inputTensorShape,
                                             const std::vector<int32_t>& axisTensorShape,
                                             const std::vector<int32_t>& outputTensorShape,
                                             const std::vector<OutputT> axisValue,
                                             tflite::TensorType outputType,
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
                                    1,
                                    flatBufferBuilder.CreateString("input"),
                                    quantizationParameters);

    auto axisTensor = CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(axisTensorShape.data(),
                                                                           axisTensorShape.size()),
                                   tflite::TensorType_INT32,
                                   2,
                                   flatBufferBuilder.CreateString("axis"));

    auto outputTensor = CreateTensor(flatBufferBuilder,
                                     flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                             outputTensorShape.size()),
                                     outputType,
                                     3,
                                     flatBufferBuilder.CreateString("output"),
                                     quantizationParameters);

    std::vector<flatbuffers::Offset<Tensor>> tensors = { inputTensor, axisTensor, outputTensor };

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(axisValue.data()),
                                                    sizeof(OutputT))));
    buffers.push_back(CreateBuffer(flatBufferBuilder));

    std::vector<int32_t> operatorInputs = {{ 0, 1 }};
    std::vector<int> subgraphInputs = {{ 0, 1 }};

    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_ArgMaxOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateArgMaxOptions(flatBufferBuilder, outputType).Union();

    if (argMinMaxOperatorCode == tflite::BuiltinOperator_ARG_MIN)
    {
        operatorBuiltinOptionsType = BuiltinOptions_ArgMinOptions;
        operatorBuiltinOptions = CreateArgMinOptions(flatBufferBuilder, outputType).Union();
    }

    // create operator
    const std::vector<int32_t> operatorOutputs{ 2 };
    flatbuffers::Offset <Operator> argMinMaxOperator =
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
                       flatBufferBuilder.CreateVector(&argMinMaxOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: ArgMinMax Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         argMinMaxOperatorCode);

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

template <typename InputT, typename OutputT>
void ArgMinMaxTest(tflite::BuiltinOperator argMinMaxOperatorCode,
                   tflite::TensorType tensorType,
                   const std::vector<armnn::BackendId>& backends,
                   const std::vector<int32_t>& inputShape,
                   const std::vector<int32_t>& axisShape,
                   std::vector<int32_t>& outputShape,
                   std::vector<InputT>& inputValues,
                   std::vector<OutputT>& expectedOutputValues,
                   OutputT axisValue,
                   tflite::TensorType outputType,
                   float quantScale = 1.0f,
                   int quantOffset  = 0)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateArgMinMaxTfLiteModel<InputT, OutputT>(argMinMaxOperatorCode,
                                                                                tensorType,
                                                                                inputShape,
                                                                                axisShape,
                                                                                outputShape,
                                                                                {axisValue},
                                                                                outputType,
                                                                                quantScale,
                                                                                quantOffset);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<InputT>(inputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<OutputT> tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<OutputT>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<InputT>(inputValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<OutputT> armnnOutputValues = armnnInterpreter.GetOutputResult<OutputT>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<OutputT>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, outputShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

} // anonymous namespace