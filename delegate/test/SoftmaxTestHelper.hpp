//
// Copyright © 2020, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn_delegate.hpp>
#include <DelegateTestInterpreter.hpp>
#include <armnnUtils/FloatingPointComparison.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/version.h>

#include <schema_generated.h>

#include <doctest/doctest.h>

namespace
{
std::vector<char> CreateSoftmaxTfLiteModel(tflite::BuiltinOperator softmaxOperatorCode,
                                           tflite::TensorType tensorType,
                                           const std::vector <int32_t>& tensorShape,
                                           float beta)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));

    std::array<flatbuffers::Offset<Tensor>, 2> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              1);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              2);

    const std::vector<int32_t> operatorInputs({0});
    const std::vector<int32_t> operatorOutputs({1});

    flatbuffers::Offset<Operator> softmaxOperator;
    flatbuffers::Offset<flatbuffers::String> modelDescription;
    flatbuffers::Offset<OperatorCode> operatorCode;

    switch (softmaxOperatorCode)
    {
        case tflite::BuiltinOperator_SOFTMAX:
            softmaxOperator =
                CreateOperator(flatBufferBuilder,
                               0,
                               flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                               flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                               BuiltinOptions_SoftmaxOptions,
                               CreateSoftmaxOptions(flatBufferBuilder, beta).Union());
                modelDescription = flatBufferBuilder.CreateString("ArmnnDelegate: Softmax Operator Model");
                operatorCode = CreateOperatorCode(flatBufferBuilder,
                                 tflite::BuiltinOperator_SOFTMAX);
            break;
        case tflite::BuiltinOperator_LOG_SOFTMAX:
            softmaxOperator =
                CreateOperator(flatBufferBuilder,
                               0,
                               flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                               flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                               BuiltinOptions_LogSoftmaxOptions,
                               CreateLogSoftmaxOptions(flatBufferBuilder).Union());
                flatBufferBuilder.CreateString("ArmnnDelegate: Log-Softmax Operator Model");
            operatorCode = CreateOperatorCode(flatBufferBuilder,
                                              tflite::BuiltinOperator_LOG_SOFTMAX);
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
                       flatBufferBuilder.CreateVector(&softmaxOperator, 1));
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

void SoftmaxTest(tflite::BuiltinOperator softmaxOperatorCode,
                 tflite::TensorType tensorType,
                 std::vector<armnn::BackendId>& backends,
                 std::vector<int32_t>& shape,
                 std::vector<float>& inputValues,
                 std::vector<float>& expectedOutputValues,
                 float beta = 0)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateSoftmaxTfLiteModel(softmaxOperatorCode,
                                                             tensorType,
                                                             shape,
                                                             beta);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<float>(inputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<float>   tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<float>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<float>(inputValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<float>   armnnOutputValues = armnnInterpreter.GetOutputResult<float>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<float>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, shape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}


/// Convenience function to run softmax and log-softmax test cases
/// \param operatorCode tflite::BuiltinOperator_SOFTMAX or tflite::BuiltinOperator_LOG_SOFTMAX
/// \param backends armnn backends to target
/// \param beta multiplicative parameter to the softmax function
/// \param expectedOutput to be checked against transformed input
void SoftmaxTestCase(tflite::BuiltinOperator operatorCode,
                     std::vector<armnn::BackendId> backends, float beta, std::vector<float> expectedOutput) {
    std::vector<float> input = {
        1.0, 2.5, 3.0, 4.5, 5.0,
        -1.0, -2.5, -3.0, -4.5, -5.0};
    std::vector<int32_t> shape = {2, 5};

    SoftmaxTest(operatorCode,
                tflite::TensorType_FLOAT32,
                backends,
                shape,
                input,
                expectedOutput,
                beta);
}

} // anonymous namespace
