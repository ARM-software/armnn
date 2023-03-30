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
std::vector<char> CreateTransposeTfLiteModel(tflite::TensorType tensorType,
                                             const std::vector <int32_t>& input0TensorShape,
                                             const std::vector <int32_t>& inputPermVecShape,
                                             const std::vector <int32_t>& outputTensorShape,
                                             const std::vector<int32_t>& inputPermVec)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;
    flatbuffers::Offset<tflite::Buffer> buffers[4]{
            CreateBuffer(flatBufferBuilder),
            CreateBuffer(flatBufferBuilder),
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(inputPermVec.data()),
                                                        sizeof(int32_t) * inputPermVec.size())),
            CreateBuffer(flatBufferBuilder)
    };
    std::array<flatbuffers::Offset<Tensor>, 3> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(input0TensorShape.data(),
                                                                      input0TensorShape.size()),
                              tensorType, 1);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputPermVecShape.data(),
                                                                      inputPermVecShape.size()),
                              tflite::TensorType_INT32, 2,
                              flatBufferBuilder.CreateString("permutation_vector"));
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,3);
    const std::vector<int32_t> operatorInputs{0, 1};
    const std::vector<int32_t> operatorOutputs{2};
    flatbuffers::Offset <Operator> transposeOperator =
            CreateOperator(flatBufferBuilder,
                           0,
                           flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                           BuiltinOptions_TransposeOptions,
                           CreateTransposeOptions(flatBufferBuilder).Union());
    const std::vector<int> subgraphInputs{0, 1};
    const std::vector<int> subgraphOutputs{2};
    flatbuffers::Offset <SubGraph> subgraph =
            CreateSubGraph(flatBufferBuilder,
                           flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                           flatBufferBuilder.CreateVector(&transposeOperator, 1));
    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: Transpose Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         tflite::BuiltinOperator_TRANSPOSE);
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
void TransposeTest(std::vector<armnn::BackendId>& backends,
                   std::vector<int32_t>& inputShape,
                   std::vector<int32_t>& inputPermVecShape,
                   std::vector<int32_t>& outputShape,
                   std::vector<T>& inputValues,
                   std::vector<int32_t>& inputPermVec,
                   std::vector<T>& expectedOutputValues)
{
    using namespace delegateTestInterpreter;

    // Create model
    std::vector<char> modelBuffer = CreateTransposeTfLiteModel(::tflite::TensorType_FLOAT32,
                                                               inputShape,
                                                               inputPermVecShape,
                                                               outputShape,
                                                               inputPermVec);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<int32_t>(inputPermVec, 1) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<int32_t>(inputPermVec, 1) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, outputShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}
}
