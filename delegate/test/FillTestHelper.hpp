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

template <typename T>
std::vector<char> CreateFillTfLiteModel(tflite::BuiltinOperator fillOperatorCode,
                                        tflite::TensorType tensorType,
                                        const std::vector<int32_t>& inputShape,
                                        const std::vector <int32_t>& tensorShape,
                                        const std::vector<T> fillValue)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(tensorShape.data()),
                                                    sizeof(int32_t) * tensorShape.size())));
    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(fillValue.data()),
                                                    sizeof(T) * fillValue.size())));
    buffers.push_back(CreateBuffer(flatBufferBuilder));

    std::array<flatbuffers::Offset<Tensor>, 3> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputShape.data(),
                                                                      inputShape.size()),
                              tflite::TensorType_INT32,
                              1,
                              flatBufferBuilder.CreateString("dims"));

    std::vector<int32_t> fillShape = {};
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(fillShape.data(),
                                                                      fillShape.size()),
                              tensorType,
                              2,
                              flatBufferBuilder.CreateString("value"));

    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              3,
                              flatBufferBuilder.CreateString("output"));

    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_FillOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateFillOptions(flatBufferBuilder).Union();

    // create operator
    const std::vector<int> operatorInputs{ {0, 1} };
    const std::vector<int> operatorOutputs{ 2 };
    flatbuffers::Offset <Operator> fillOperator =
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
                       flatBufferBuilder.CreateVector(&fillOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Fill Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         fillOperatorCode);

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
void FillTest(tflite::BuiltinOperator fillOperatorCode,
              tflite::TensorType tensorType,
              const std::vector<armnn::BackendId>& backends,
              std::vector<int32_t >& inputShape,
              std::vector<int32_t >& tensorShape,
              std::vector<T>& expectedOutputValues,
              T fillValue)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateFillTfLiteModel<T>(fillOperatorCode,
                                                             tensorType,
                                                             inputShape,
                                                             tensorShape,
                                                             {fillValue});

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, tensorShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

} // anonymous namespace
