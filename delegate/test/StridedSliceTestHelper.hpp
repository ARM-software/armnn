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

std::vector<char> CreateStridedSliceTfLiteModel(tflite::TensorType tensorType,
                                                const std::vector<int32_t>& inputTensorShape,
                                                const std::vector<int32_t>& beginTensorData,
                                                const std::vector<int32_t>& endTensorData,
                                                const std::vector<int32_t>& strideTensorData,
                                                const std::vector<int32_t>& beginTensorShape,
                                                const std::vector<int32_t>& endTensorShape,
                                                const std::vector<int32_t>& strideTensorShape,
                                                const std::vector<int32_t>& outputTensorShape,
                                                const int32_t beginMask,
                                                const int32_t endMask,
                                                const int32_t ellipsisMask,
                                                const int32_t newAxisMask,
                                                const int32_t ShrinkAxisMask,
                                                const armnn::DataLayout& dataLayout)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    flatbuffers::Offset<tflite::Buffer> buffers[6] = {
            CreateBuffer(flatBufferBuilder),
            CreateBuffer(flatBufferBuilder),
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(beginTensorData.data()),
                                                        sizeof(int32_t) * beginTensorData.size())),
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(endTensorData.data()),
                                                        sizeof(int32_t) * endTensorData.size())),
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(strideTensorData.data()),
                                                        sizeof(int32_t) * strideTensorData.size())),
            CreateBuffer(flatBufferBuilder)
    };

    std::array<flatbuffers::Offset<Tensor>, 5> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              tensorType,
                              1,
                              flatBufferBuilder.CreateString("input"));
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(beginTensorShape.data(),
                                                                      beginTensorShape.size()),
                              ::tflite::TensorType_INT32,
                              2,
                              flatBufferBuilder.CreateString("begin_tensor"));
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(endTensorShape.data(),
                                                                      endTensorShape.size()),
                              ::tflite::TensorType_INT32,
                              3,
                              flatBufferBuilder.CreateString("end_tensor"));
    tensors[3] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(strideTensorShape.data(),
                                                                      strideTensorShape.size()),
                              ::tflite::TensorType_INT32,
                              4,
                              flatBufferBuilder.CreateString("stride_tensor"));
    tensors[4] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              5,
                              flatBufferBuilder.CreateString("output"));


    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_StridedSliceOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateStridedSliceOptions(flatBufferBuilder,
                                                                                 beginMask,
                                                                                 endMask,
                                                                                 ellipsisMask,
                                                                                 newAxisMask,
                                                                                 ShrinkAxisMask).Union();

    const std::vector<int> operatorInputs{ 0, 1, 2, 3 };
    const std::vector<int> operatorOutputs{ 4 };
    flatbuffers::Offset <Operator> sliceOperator =
            CreateOperator(flatBufferBuilder,
                           0,
                           flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                           operatorBuiltinOptionsType,
                           operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{ 0, 1, 2, 3 };
    const std::vector<int> subgraphOutputs{ 4 };
    flatbuffers::Offset <SubGraph> subgraph =
            CreateSubGraph(flatBufferBuilder,
                           flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                           flatBufferBuilder.CreateVector(&sliceOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: StridedSlice Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         BuiltinOperator_STRIDED_SLICE);

    flatbuffers::Offset <Model> flatbufferModel =
            CreateModel(flatBufferBuilder,
                        TFLITE_SCHEMA_VERSION,
                        flatBufferBuilder.CreateVector(&operatorCode, 1),
                        flatBufferBuilder.CreateVector(&subgraph, 1),
                        modelDescription,
                        flatBufferBuilder.CreateVector(buffers, 6));

    flatBufferBuilder.Finish(flatbufferModel, armnnDelegate::FILE_IDENTIFIER);

    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

template <typename T>
void StridedSliceTestImpl(std::vector<armnn::BackendId>& backends,
                          std::vector<T>& inputValues,
                          std::vector<T>& expectedOutputValues,
                          std::vector<int32_t>& beginTensorData,
                          std::vector<int32_t>& endTensorData,
                          std::vector<int32_t>& strideTensorData,
                          std::vector<int32_t>& inputTensorShape,
                          std::vector<int32_t>& beginTensorShape,
                          std::vector<int32_t>& endTensorShape,
                          std::vector<int32_t>& strideTensorShape,
                          std::vector<int32_t>& outputTensorShape,
                          const int32_t beginMask = 0,
                          const int32_t endMask = 0,
                          const int32_t ellipsisMask = 0,
                          const int32_t newAxisMask = 0,
                          const int32_t ShrinkAxisMask = 0,
                          const armnn::DataLayout& dataLayout = armnn::DataLayout::NHWC)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateStridedSliceTfLiteModel(
            ::tflite::TensorType_FLOAT32,
            inputTensorShape,
            beginTensorData,
            endTensorData,
            strideTensorData,
            beginTensorShape,
            endTensorShape,
            strideTensorShape,
            outputTensorShape,
            beginMask,
            endMask,
            ellipsisMask,
            newAxisMask,
            ShrinkAxisMask,
            dataLayout);

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
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, outputTensorShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
} // End of StridedSlice Test

} // anonymous namespace