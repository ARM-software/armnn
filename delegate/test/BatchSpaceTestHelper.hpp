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

std::vector<char> CreateBatchSpaceTfLiteModel(tflite::BuiltinOperator batchSpaceOperatorCode,
                                              tflite::TensorType tensorType,
                                              std::vector<int32_t>& inputTensorShape,
                                              std::vector <int32_t>& outputTensorShape,
                                              std::vector<unsigned int>& blockData,
                                              std::vector<std::pair<unsigned int, unsigned int>>& cropsPadData,
                                              float quantScale = 1.0f,
                                              int quantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::array<flatbuffers::Offset<tflite::Buffer>, 5> buffers;
    buffers[0] = CreateBuffer(flatBufferBuilder);
    buffers[1] = CreateBuffer(flatBufferBuilder);
    buffers[2] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(blockData.data()),
                                                                  sizeof(int32_t) * blockData.size()));
    buffers[3] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(cropsPadData.data()),
                                                                  sizeof(int64_t) * cropsPadData.size()));
    buffers[4] = CreateBuffer(flatBufferBuilder);

    auto quantizationParameters =
            CreateQuantizationParameters(flatBufferBuilder,
                                         0,
                                         0,
                                         flatBufferBuilder.CreateVector<float>({ quantScale }),
                                         flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));

    std::string cropsOrPadding =
            batchSpaceOperatorCode == tflite::BuiltinOperator_BATCH_TO_SPACE_ND ? "crops" : "padding";

    std::vector<int32_t> blockShape { 2 };
    std::vector<int32_t> cropsOrPaddingShape { 2, 2 };

    std::array<flatbuffers::Offset<Tensor>, 4> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              tensorType,
                              1,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);

    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(blockShape.data(),
                                                                      blockShape.size()),
                              ::tflite::TensorType_INT32,
                              2,
                              flatBufferBuilder.CreateString("block"),
                              quantizationParameters);

    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(cropsOrPaddingShape.data(),
                                                                      cropsOrPaddingShape.size()),
                              ::tflite::TensorType_INT32,
                              3,
                              flatBufferBuilder.CreateString(cropsOrPadding),
                              quantizationParameters);

    // Create output tensor
    tensors[3] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              4,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    // Create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_NONE;
    flatbuffers::Offset<void> operatorBuiltinOptions = 0;
    switch (batchSpaceOperatorCode)
    {
        case tflite::BuiltinOperator_BATCH_TO_SPACE_ND:
        {
            operatorBuiltinOptionsType = tflite::BuiltinOptions_BatchToSpaceNDOptions;
            operatorBuiltinOptions = CreateBatchToSpaceNDOptions(flatBufferBuilder).Union();
            break;
        }
        case tflite::BuiltinOperator_SPACE_TO_BATCH_ND:
        {
            operatorBuiltinOptionsType = tflite::BuiltinOptions_SpaceToBatchNDOptions;
            operatorBuiltinOptions = CreateSpaceToBatchNDOptions(flatBufferBuilder).Union();
            break;
        }
        default:
            break;
    }

    const std::vector<int> operatorInputs{ {0, 1, 2} };
    const std::vector<int> operatorOutputs{ 3 };
    flatbuffers::Offset <Operator> batchSpaceOperator =
            CreateOperator(flatBufferBuilder,
                           0,
                           flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                           operatorBuiltinOptionsType,
                           operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{ {0, 1, 2} };
    const std::vector<int> subgraphOutputs{ 3 };
    flatbuffers::Offset <SubGraph> subgraph =
            CreateSubGraph(flatBufferBuilder,
                           flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                           flatBufferBuilder.CreateVector(&batchSpaceOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: BatchSpace Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, batchSpaceOperatorCode);

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
void BatchSpaceTest(tflite::BuiltinOperator controlOperatorCode,
                    tflite::TensorType tensorType,
                    std::vector<armnn::BackendId>& backends,
                    std::vector<int32_t>& inputShape,
                    std::vector<int32_t>& expectedOutputShape,
                    std::vector<T>& inputValues,
                    std::vector<unsigned int>& blockShapeValues,
                    std::vector<std::pair<unsigned int, unsigned int>>& cropsPaddingValues,
                    std::vector<T>& expectedOutputValues,
                    float quantScale = 1.0f,
                    int quantOffset  = 0)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateBatchSpaceTfLiteModel(controlOperatorCode,
                                                                tensorType,
                                                                inputShape,
                                                                expectedOutputShape,
                                                                blockShapeValues,
                                                                cropsPaddingValues,
                                                                quantScale,
                                                                quantOffset);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor(inputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor(inputValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, expectedOutputShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

} // anonymous namespace