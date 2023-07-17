//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
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
    std::vector<char> CreateReverseV2TfLiteModel(tflite::BuiltinOperator operatorCode,
                                                 tflite::TensorType inputTensorType,
                                                 const std::vector <int32_t>& inputTensorShape,
                                                 const std::vector <int32_t>& axisTensorData,
                                                 const std::vector <int32_t>& axisTensorShape,
                                                 const std::vector <int32_t>& outputTensorShape)
    {
        using namespace tflite;
        flatbuffers::FlatBufferBuilder flatBufferBuilder;

        std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
        buffers.push_back(CreateBuffer(flatBufferBuilder));
        buffers.push_back(CreateBuffer(flatBufferBuilder));
        buffers.push_back(CreateBuffer(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector(
                                               reinterpret_cast<const uint8_t*>(axisTensorData.data()),
                                               sizeof(int32_t) * axisTensorData.size())));
        buffers.push_back(CreateBuffer(flatBufferBuilder));

        std::array<flatbuffers::Offset<Tensor>, 3> tensors;
        tensors[0] = CreateTensor(flatBufferBuilder,
                                  flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                          inputTensorShape.size()),
                                  inputTensorType,
                                  1,
                                  flatBufferBuilder.CreateString("input_tensor"));

        tensors[1] = CreateTensor(flatBufferBuilder,
                                  flatBufferBuilder.CreateVector<int32_t>(axisTensorShape.data(),
                                                                          axisTensorShape.size()),
                                  TensorType_INT32,
                                  2,
                                  flatBufferBuilder.CreateString("axis_input_tensor"));

        tensors[2] = CreateTensor(flatBufferBuilder,
                                  flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                          outputTensorShape.size()),
                                  inputTensorType,
                                  3,
                                  flatBufferBuilder.CreateString("output_tensor"));

        // Create Operator
        tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_NONE;
        flatbuffers::Offset<void> operatorBuiltinOption = 0;

        const std::vector<int> operatorInputs{0, 1};
        const std::vector<int> operatorOutputs{2};
        flatbuffers::Offset <Operator> reverseV2Operator =
                CreateOperator(flatBufferBuilder,
                               0,
                               flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                               flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                               operatorBuiltinOptionsType,
                               operatorBuiltinOption);

        const std::vector<int> subgraphInputs{0, 1};
        const std::vector<int> subgraphOutputs{2};
        flatbuffers::Offset <SubGraph> subgraph =
                CreateSubGraph(flatBufferBuilder,
                               flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                               flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                               flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                               flatBufferBuilder.CreateVector(&reverseV2Operator, 1));

        flatbuffers::Offset <flatbuffers::String> modelDescription =
                flatBufferBuilder.CreateString("ArmnnDelegate: ReverseV2 Operator Model");
        flatbuffers::Offset <OperatorCode> opCode = CreateOperatorCode(flatBufferBuilder, operatorCode);

        flatbuffers::Offset <Model> flatbufferModel =
                CreateModel(flatBufferBuilder,
                            TFLITE_SCHEMA_VERSION,
                            flatBufferBuilder.CreateVector(&opCode, 1),
                            flatBufferBuilder.CreateVector(&subgraph, 1),
                            modelDescription,
                            flatBufferBuilder.CreateVector(buffers.data(), buffers.size()));

        flatBufferBuilder.Finish(flatbufferModel, armnnDelegate::FILE_IDENTIFIER);

        return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                                 flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
    }

    void ReverseV2FP32TestImpl(tflite::BuiltinOperator operatorCode,
                               std::vector<armnn::BackendId>& backends,
                               std::vector<float>& inputValues,
                               std::vector<int32_t> inputShape,
                               std::vector<int32_t> axisValues,
                               std::vector<int32_t> axisShapeDims,
                               std::vector<float>& expectedOutputValues,
                               std::vector<int32_t> expectedOutputShape)
    {
        using namespace delegateTestInterpreter;

        std::vector<char> modelBuffer = CreateReverseV2TfLiteModel(operatorCode,
                                                                   ::tflite::TensorType_FLOAT32,
                                                                   inputShape,
                                                                   axisValues,
                                                                   axisShapeDims,
                                                                   expectedOutputShape);

        // Setup interpreter with just TFLite Runtime.
        auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
        CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
        CHECK(tfLiteInterpreter.FillInputTensor<float>(inputValues, 0) == kTfLiteOk);
        CHECK(tfLiteInterpreter.FillInputTensor<int32_t>(axisValues, 1) == kTfLiteOk);
        CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
        std::vector<float>   tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<float>(0);
        std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

        // Setup interpreter with Arm NN Delegate applied.
        auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
        CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
        CHECK(armnnInterpreter.FillInputTensor<float>(inputValues, 0) == kTfLiteOk);
        CHECK(armnnInterpreter.FillInputTensor<int32_t>(axisValues, 1) == kTfLiteOk);
        CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
        std::vector<float>   armnnOutputValues = armnnInterpreter.GetOutputResult<float>(0);
        std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

        armnnDelegate::CompareOutputData<float>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
        armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, expectedOutputShape);

        tfLiteInterpreter.Cleanup();
        armnnInterpreter.Cleanup();
    }

} // anonymous namespace
