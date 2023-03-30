//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
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
std::vector<char> CreateBatchMatMulTfLiteModel(
        tflite::BuiltinOperator bmmOperatorCode,
        tflite::TensorType tensorType,
        const std::vector <int32_t>& LHSInputTensorShape,
        const std::vector <int32_t>& RHSInputTensorShape,
        const std::vector <int32_t>& outputTensorShape,
        bool adjX = false,
        bool adjY = false,
        float quantScale = 1.0f,
        int quantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));

    auto quantizationParameters =
            CreateQuantizationParameters(flatBufferBuilder,
                                         0,
                                         0,
                                         flatBufferBuilder.CreateVector<float>({ quantScale }),
                                         flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));

    std::array<flatbuffers::Offset<Tensor>, 3> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(LHSInputTensorShape.data(),
                                                                      LHSInputTensorShape.size()),
                              tensorType,
                              1,
                              flatBufferBuilder.CreateString("LHSInput"),
                              quantizationParameters);

    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(RHSInputTensorShape.data(),
                                                                      RHSInputTensorShape.size()),
                              tensorType,
                              2,
                              flatBufferBuilder.CreateString("RHSInput"),
                              quantizationParameters);

    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              3,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_BatchMatMulOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateBatchMatMulOptions(flatBufferBuilder,
                                                                                adjX,
                                                                                adjY).Union();

    const std::vector<int32_t> operatorInputs{{0, 1}};
    const std::vector<int32_t> operatorOutputs{2};
    flatbuffers::Offset <Operator> bmmOperator =
            CreateOperator(flatBufferBuilder,
                           0,
                           flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(),
                                                                   operatorOutputs.size()),
                           operatorBuiltinOptionsType,
                           operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{{0, 1}};
    const std::vector<int> subgraphOutputs{2};
    flatbuffers::Offset <SubGraph> subgraph =
            CreateSubGraph(flatBufferBuilder,
                           flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(),
                                                                   subgraphOutputs.size()),
                           flatBufferBuilder.CreateVector(&bmmOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: BatchMatMul Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, bmmOperatorCode);

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
void BatchMatMulTest(tflite::BuiltinOperator bmmOperatorCode,
                   tflite::TensorType tensorType,
                   std::vector<armnn::BackendId>& backends,
                   std::vector<int32_t>& LHSInputShape,
                   std::vector<int32_t>& RHSInputShape,
                   std::vector<int32_t>& outputShape,
                   std::vector<T>& LHSInputValues,
                   std::vector<T>& RHSInputValues,
                   std::vector<T>& expectedOutputValues,
                   bool adjX = false,
                   bool adjY = false,
                   float quantScale = 1.0f,
                   int quantOffset  = 0)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateBatchMatMulTfLiteModel(bmmOperatorCode,
                                                                 tensorType,
                                                                 LHSInputShape,
                                                                 RHSInputShape,
                                                                 outputShape,
                                                                 adjX,
                                                                 adjY,
                                                                 quantScale,
                                                                 quantOffset);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(LHSInputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(RHSInputValues, 1) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(LHSInputValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(RHSInputValues, 1) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, outputShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

} // anonymous namespace




