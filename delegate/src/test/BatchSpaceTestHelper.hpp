//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

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

    std::array<flatbuffers::Offset<tflite::Buffer>, 3> buffers;
    buffers[0] = CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({}));
    buffers[1] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(blockData.data()),
                                                                  sizeof(int32_t) * blockData.size()));
    buffers[2] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(cropsPadData.data()),
                                                                  sizeof(int64_t) * cropsPadData.size()));

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
                              0,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);

    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(blockShape.data(),
                                                                      blockShape.size()),
                              ::tflite::TensorType_INT32,
                              1,
                              flatBufferBuilder.CreateString("block"),
                              quantizationParameters);

    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(cropsOrPaddingShape.data(),
                                                                      cropsOrPaddingShape.size()),
                              ::tflite::TensorType_INT32,
                              2,
                              flatBufferBuilder.CreateString(cropsOrPadding),
                              quantizationParameters);

    // Create output tensor
    tensors[3] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              0,
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

    flatBufferBuilder.Finish(flatbufferModel);

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
    using namespace tflite;
    std::vector<char> modelBuffer = CreateBatchSpaceTfLiteModel(controlOperatorCode,
                                                                tensorType,
                                                                inputShape,
                                                                expectedOutputShape,
                                                                blockShapeValues,
                                                                cropsPaddingValues,
                                                                quantScale,
                                                                quantOffset);

    const Model* tfLiteModel = GetModel(modelBuffer.data());

    // Create TfLite Interpreters
    std::unique_ptr<Interpreter> armnnDelegateInterpreter;
    CHECK(InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
                  (&armnnDelegateInterpreter) == kTfLiteOk);
    CHECK(armnnDelegateInterpreter != nullptr);
    CHECK(armnnDelegateInterpreter->AllocateTensors() == kTfLiteOk);

    std::unique_ptr<Interpreter> tfLiteInterpreter;
    CHECK(InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
                  (&tfLiteInterpreter) == kTfLiteOk);
    CHECK(tfLiteInterpreter != nullptr);
    CHECK(tfLiteInterpreter->AllocateTensors() == kTfLiteOk);

    // Create the ArmNN Delegate
    armnnDelegate::DelegateOptions delegateOptions(backends);
    std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
            theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                             armnnDelegate::TfLiteArmnnDelegateDelete);
    CHECK(theArmnnDelegate != nullptr);

    // Modify armnnDelegateInterpreter to use armnnDelegate
    CHECK(armnnDelegateInterpreter->ModifyGraphWithDelegate(theArmnnDelegate.get()) == kTfLiteOk);

    // Set input data
    armnnDelegate::FillInput<T>(tfLiteInterpreter, 0, inputValues);
    armnnDelegate::FillInput<T>(armnnDelegateInterpreter, 0, inputValues);

    // Run EnqueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    // Compare output data
    armnnDelegate::CompareOutputData<T>(tfLiteInterpreter,
                                        armnnDelegateInterpreter,
                                        expectedOutputShape,
                                        expectedOutputValues);

    armnnDelegateInterpreter.reset(nullptr);
    tfLiteInterpreter.reset(nullptr);
}

} // anonymous namespace