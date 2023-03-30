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

std::vector<char> CreateReduceTfLiteModel(tflite::BuiltinOperator reduceOperatorCode,
                                          tflite::TensorType tensorType,
                                          std::vector<int32_t>& input0TensorShape,
                                          std::vector<int32_t>& input1TensorShape,
                                          const std::vector <int32_t>& outputTensorShape,
                                          std::vector<int32_t>& axisData,
                                          const bool keepDims,
                                          float quantScale = 1.0f,
                                          int quantOffset  = 0,
                                          bool kTfLiteNoQuantizationForQuantized = false)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    flatbuffers::Offset<tflite::Buffer> buffers[4] = {
            CreateBuffer(flatBufferBuilder),
            CreateBuffer(flatBufferBuilder),
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(axisData.data()),
                                                        sizeof(int32_t) * axisData.size())),
            CreateBuffer(flatBufferBuilder)
    };

    flatbuffers::Offset<tflite::QuantizationParameters> quantizationParametersAxis
            = CreateQuantizationParameters(flatBufferBuilder);

    flatbuffers::Offset<tflite::QuantizationParameters> quantizationParameters;

    if (kTfLiteNoQuantizationForQuantized)
    {
        if ((quantScale == 1 || quantScale == 0) && quantOffset == 0)
        {
            // Creates quantization parameter with quantization.type = kTfLiteNoQuantization
            quantizationParameters = CreateQuantizationParameters(flatBufferBuilder);
        }
        else
        {
            // Creates quantization parameter with quantization.type != kTfLiteNoQuantization
            quantizationParameters = CreateQuantizationParameters(
                    flatBufferBuilder,
                    0,
                    0,
                    flatBufferBuilder.CreateVector<float>({quantScale}),
                    flatBufferBuilder.CreateVector<int64_t>({quantOffset}));
        }
    }
    else
    {
        quantizationParameters = CreateQuantizationParameters(
                flatBufferBuilder,
                0,
                0,
                flatBufferBuilder.CreateVector<float>({quantScale}),
                flatBufferBuilder.CreateVector<int64_t>({quantOffset}));
    }

    std::array<flatbuffers::Offset<Tensor>, 3> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(input0TensorShape.data(),
                                                                      input0TensorShape.size()),
                              tensorType,
                              1,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);

    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(input1TensorShape.data(),
                                                                      input1TensorShape.size()),
                              ::tflite::TensorType_INT32,
                              2,
                              flatBufferBuilder.CreateString("axis"),
                              quantizationParametersAxis);

    // Create output tensor
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              3,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    // Create operator. Reduce operations MIN, MAX, SUM, MEAN, PROD uses ReducerOptions.
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_ReducerOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateReducerOptions(flatBufferBuilder, keepDims).Union();

    const std::vector<int> operatorInputs{ {0, 1} };
    const std::vector<int> operatorOutputs{ 2 };
    flatbuffers::Offset <Operator> reduceOperator =
            CreateOperator(flatBufferBuilder,
                           0,
                           flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                           operatorBuiltinOptionsType,
                           operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{ {0, 1} };
    const std::vector<int> subgraphOutputs{ 2 };
    flatbuffers::Offset <SubGraph> subgraph =
            CreateSubGraph(flatBufferBuilder,
                           flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                           flatBufferBuilder.CreateVector(&reduceOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: Reduce Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, reduceOperatorCode);

    flatbuffers::Offset <Model> flatbufferModel =
            CreateModel(flatBufferBuilder,
                        TFLITE_SCHEMA_VERSION,
                        flatBufferBuilder.CreateVector(&operatorCode, 1),
                        flatBufferBuilder.CreateVector(&subgraph, 1),
                        modelDescription,
                        flatBufferBuilder.CreateVector(buffers, 4));

    flatBufferBuilder.Finish(flatbufferModel, armnnDelegate::FILE_IDENTIFIER);

    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

template <typename T>
void ReduceTest(tflite::BuiltinOperator reduceOperatorCode,
                tflite::TensorType tensorType,
                std::vector<armnn::BackendId>& backends,
                std::vector<int32_t>& input0Shape,
                std::vector<int32_t>& input1Shape,
                std::vector<int32_t>& expectedOutputShape,
                std::vector<T>& input0Values,
                std::vector<int32_t>& input1Values,
                std::vector<T>& expectedOutputValues,
                const bool keepDims,
                float quantScale = 1.0f,
                int quantOffset  = 0)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBufferArmNN = CreateReduceTfLiteModel(reduceOperatorCode,
                                                                 tensorType,
                                                                 input0Shape,
                                                                 input1Shape,
                                                                 expectedOutputShape,
                                                                 input1Values,
                                                                 keepDims,
                                                                 quantScale,
                                                                 quantOffset,
                                                                 false);
    std::vector<char> modelBufferTFLite = CreateReduceTfLiteModel(reduceOperatorCode,
                                                                  tensorType,
                                                                  input0Shape,
                                                                  input1Shape,
                                                                  expectedOutputShape,
                                                                  input1Values,
                                                                  keepDims,
                                                                  quantScale,
                                                                  quantOffset,
                                                                  true);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBufferTFLite);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(input0Values, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBufferArmNN, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(input0Values, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, expectedOutputShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

} // anonymous namespace