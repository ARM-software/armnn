//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn_delegate.hpp>
#include <DelegateTestInterpreter.hpp>

#include <tensorflow/lite/version.h>

namespace
{
    std::vector<char> CreateBroadcastToTfLiteModel(tflite::BuiltinOperator operatorCode,
                                                   tflite::TensorType inputTensorType,
                                                   const std::vector<int32_t>& inputTensorShape,
                                                   const std::vector<int32_t>& shapeTensorShape,
                                                   const std::vector<int32_t>& shapeTensorData,
                                                   const std::vector<int32_t>& outputTensorShape)
    {
        using namespace tflite;
        flatbuffers::FlatBufferBuilder flatBufferBuilder;

        std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
        buffers.push_back(CreateBuffer(flatBufferBuilder));
        buffers.push_back(CreateBuffer(flatBufferBuilder));
        buffers.push_back(CreateBuffer(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector(
                                       reinterpret_cast<const uint8_t*>(shapeTensorData.data()),
                                       sizeof(int32_t) * shapeTensorData.size())));
        buffers.push_back(CreateBuffer(flatBufferBuilder));

        float   qScale  = 1.0f;
        int32_t qOffset = 0;

        auto quantizationParameters =
                CreateQuantizationParameters(flatBufferBuilder,
                                             0,
                                             0,
                                             flatBufferBuilder.CreateVector<float>({ qScale }),
                                             flatBufferBuilder.CreateVector<int64_t>({ qOffset }));

        std::array<flatbuffers::Offset<Tensor>, 3> tensors;
        tensors[0] = CreateTensor(flatBufferBuilder,
                                  flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                          inputTensorShape.size()),
                                  inputTensorType,
                                  1,
                                  flatBufferBuilder.CreateString("input_tensor"),
                                  quantizationParameters);

        tensors[1] = CreateTensor(flatBufferBuilder,
                                  flatBufferBuilder.CreateVector<int32_t>(shapeTensorShape.data(),
                                                                          shapeTensorShape.size()),
                                  TensorType_INT32,
                                  2,
                                  flatBufferBuilder.CreateString("shape_input_tensor"),
                                  quantizationParameters);

        tensors[2] = CreateTensor(flatBufferBuilder,
                                  flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                          outputTensorShape.size()),
                                  inputTensorType,
                                  3,
                                  flatBufferBuilder.CreateString("output_tensor"),
                                  quantizationParameters);

        // Create Operator
        tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_BroadcastToOptions;
        flatbuffers::Offset<void> operatorBuiltinOption = 0;

        const std::vector<int> operatorInputs {0, 1};
        const std::vector<int> operatorOutputs {2};

        flatbuffers::Offset<Operator> broadcastOperator =
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
                               flatBufferBuilder.CreateVector(&broadcastOperator, 1));

        flatbuffers::Offset <flatbuffers::String> modelDescription =
                flatBufferBuilder.CreateString("ArmnnDelegate: BrodacastTo Operator Model");
        flatbuffers::Offset <OperatorCode> opCode = CreateOperatorCode(flatBufferBuilder,0,
                                                                       0, 2,
                                                                       tflite::BuiltinOperator_BROADCAST_TO);

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

    template<typename T>
    void BroadcastToTestImpl(tflite::TensorType inputTensorType,
                             tflite::BuiltinOperator operatorCode,
                             std::vector<T>& inputValues,
                             std::vector<int32_t> inputShape,
                             std::vector<int32_t> shapeShapes,
                             std::vector<int32_t> shapeData,
                             std::vector<T>& expectedOutputValues,
                             std::vector<int32_t> expectedOutputShape,
                             const std::vector<armnn::BackendId>& backends)
    {
        using namespace delegateTestInterpreter;

        std::vector<char> modelBuffer = CreateBroadcastToTfLiteModel(operatorCode,
                                                                     inputTensorType,
                                                                     inputShape,
                                                                     shapeShapes,
                                                                     shapeData,
                                                                     expectedOutputShape);


        // Setup interpreter with just TFLite Runtime.
        auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
        CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
        CHECK(tfLiteInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
        CHECK(tfLiteInterpreter.FillInputTensor<int32_t>(shapeData, 1) == kTfLiteOk);
        CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
        std::vector<T>   tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
        std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

        // Setup interpreter with Arm NN Delegate applied.
        auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, CaptureAvailableBackends(backends));
        CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
        CHECK(armnnInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
        CHECK(armnnInterpreter.FillInputTensor<int32_t>(shapeData, 1) == kTfLiteOk);
        CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
        std::vector<T>   armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
        std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

        armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
        armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, expectedOutputShape);

        tfLiteInterpreter.Cleanup();
        armnnInterpreter.Cleanup();
    }

} // anonymous namespace