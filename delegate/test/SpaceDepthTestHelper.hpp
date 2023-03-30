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
std::vector<char> CreateSpaceDepthTfLiteModel(tflite::BuiltinOperator spaceDepthOperatorCode,
                                              tflite::TensorType tensorType,
                                              const std::vector <int32_t>& inputTensorShape,
                                              const std::vector <int32_t>& outputTensorShape,
                                              int32_t blockSize)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    auto quantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({  1.0f }),
                                     flatBufferBuilder.CreateVector<int64_t>({ 0 }));

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));

    std::array<flatbuffers::Offset<Tensor>, 2> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              tensorType,
                              1,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              2,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    const std::vector<int32_t> operatorInputs({0});
    const std::vector<int32_t> operatorOutputs({1});

    flatbuffers::Offset<Operator> spaceDepthOperator;
    flatbuffers::Offset<flatbuffers::String> modelDescription;
    flatbuffers::Offset<OperatorCode> operatorCode;

    switch (spaceDepthOperatorCode)
    {
        case tflite::BuiltinOperator_SPACE_TO_DEPTH:
            spaceDepthOperator =
                CreateOperator(flatBufferBuilder,
                               0,
                               flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                               flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                               BuiltinOptions_SpaceToDepthOptions,
                               CreateSpaceToDepthOptions(flatBufferBuilder, blockSize).Union());
                modelDescription = flatBufferBuilder.CreateString("ArmnnDelegate: SPACE_TO_DEPTH Operator Model");
                operatorCode = CreateOperatorCode(flatBufferBuilder,
                                 tflite::BuiltinOperator_SPACE_TO_DEPTH);
            break;
        case tflite::BuiltinOperator_DEPTH_TO_SPACE:
            spaceDepthOperator =
                CreateOperator(flatBufferBuilder,
                               0,
                               flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                               flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                               BuiltinOptions_DepthToSpaceOptions,
                               CreateDepthToSpaceOptions(flatBufferBuilder, blockSize).Union());
                flatBufferBuilder.CreateString("ArmnnDelegate: DEPTH_TO_SPACE Operator Model");
            operatorCode = CreateOperatorCode(flatBufferBuilder,
                                              tflite::BuiltinOperator_DEPTH_TO_SPACE);
            break;
        default:
            break;
    }
    const std::vector<int32_t> subgraphInputs({0});
    const std::vector<int32_t> subgraphOutputs({1});
    flatbuffers::Offset<SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&spaceDepthOperator, 1));
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

template <typename T>
void SpaceDepthTest(tflite::BuiltinOperator spaceDepthOperatorCode,
                    tflite::TensorType tensorType,
                    std::vector<armnn::BackendId>& backends,
                    std::vector<int32_t>& inputShape,
                    std::vector<int32_t>& outputShape,
                    std::vector<T>& inputValues,
                    std::vector<T>& expectedOutputValues,
                    int32_t blockSize = 2)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateSpaceDepthTfLiteModel(spaceDepthOperatorCode,
                                                                tensorType,
                                                                inputShape,
                                                                outputShape,
                                                                blockSize);

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
