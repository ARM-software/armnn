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

struct StreamRedirector
{
public:
    StreamRedirector(std::ostream &stream, std::streambuf *newStreamBuffer)
        : m_Stream(stream), m_BackupBuffer(m_Stream.rdbuf(newStreamBuffer)) {}

    ~StreamRedirector() { m_Stream.rdbuf(m_BackupBuffer); }

private:
    std::ostream &m_Stream;
    std::streambuf *m_BackupBuffer;
};

std::vector<char> CreateAddDivTfLiteModel(tflite::TensorType tensorType,
                                          const std::vector<int32_t>& tensorShape,
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
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));

    auto quantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({ quantScale }),
                                     flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));


    std::array<flatbuffers::Offset<Tensor>, 5> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              1,
                              flatBufferBuilder.CreateString("input_0"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              2,
                              flatBufferBuilder.CreateString("input_1"),
                              quantizationParameters);
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              3,
                              flatBufferBuilder.CreateString("input_2"),
                              quantizationParameters);
    tensors[3] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              4,
                              flatBufferBuilder.CreateString("add"),
                              quantizationParameters);
    tensors[4] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              5,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    // create operator
    tflite::BuiltinOptions addBuiltinOptionsType = tflite::BuiltinOptions_AddOptions;
    flatbuffers::Offset<void> addBuiltinOptions =
        CreateAddOptions(flatBufferBuilder, ActivationFunctionType_NONE).Union();

    tflite::BuiltinOptions divBuiltinOptionsType = tflite::BuiltinOptions_DivOptions;
    flatbuffers::Offset<void> divBuiltinOptions =
        CreateAddOptions(flatBufferBuilder, ActivationFunctionType_NONE).Union();

    std::array<flatbuffers::Offset<Operator>, 2> operators;
    const std::vector<int32_t> addInputs{0, 1};
    const std::vector<int32_t> addOutputs{3};
    operators[0] = CreateOperator(flatBufferBuilder,
                                  0,
                                  flatBufferBuilder.CreateVector<int32_t>(addInputs.data(), addInputs.size()),
                                  flatBufferBuilder.CreateVector<int32_t>(addOutputs.data(), addOutputs.size()),
                                  addBuiltinOptionsType,
                                  addBuiltinOptions);
    const std::vector<int32_t> divInputs{3, 2};
    const std::vector<int32_t> divOutputs{4};
    operators[1] = CreateOperator(flatBufferBuilder,
                                  1,
                                  flatBufferBuilder.CreateVector<int32_t>(divInputs.data(), divInputs.size()),
                                  flatBufferBuilder.CreateVector<int32_t>(divOutputs.data(), divOutputs.size()),
                                  divBuiltinOptionsType,
                                  divBuiltinOptions);

    const std::vector<int> subgraphInputs{0, 1, 2};
    const std::vector<int> subgraphOutputs{4};
    flatbuffers::Offset<SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(operators.data(), operators.size()));

    flatbuffers::Offset<flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Add and Div Operator Model");

    std::array<flatbuffers::Offset<OperatorCode>, 2> codes;
    codes[0] = CreateOperatorCode(flatBufferBuilder, tflite::BuiltinOperator_ADD);
    codes[1] = CreateOperatorCode(flatBufferBuilder, tflite::BuiltinOperator_DIV);

    flatbuffers::Offset<Model> flatbufferModel =
        CreateModel(flatBufferBuilder,
                    TFLITE_SCHEMA_VERSION,
                    flatBufferBuilder.CreateVector(codes.data(), codes.size()),
                    flatBufferBuilder.CreateVector(&subgraph, 1),
                    modelDescription,
                    flatBufferBuilder.CreateVector(buffers.data(), buffers.size()));

    flatBufferBuilder.Finish(flatbufferModel, armnnDelegate::FILE_IDENTIFIER);

    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

std::vector<char> CreateCosTfLiteModel(tflite::TensorType tensorType,
                                       const std::vector <int32_t>& tensorShape,
                                       float quantScale = 1.0f,
                                       int quantOffset = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder));

    auto quantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({quantScale}),
                                     flatBufferBuilder.CreateVector<int64_t>({quantOffset}));

    std::array<flatbuffers::Offset<Tensor>, 2> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    const std::vector<int32_t> operatorInputs({0});
    const std::vector<int32_t> operatorOutputs({1});

    flatbuffers::Offset<Operator> ceilOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       BuiltinOptions_NONE);

    flatbuffers::Offset<flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: CEIL Operator Model");
    flatbuffers::Offset<OperatorCode> operatorCode =
        CreateOperatorCode(flatBufferBuilder, tflite::BuiltinOperator_COS);

    const std::vector<int32_t> subgraphInputs({0});
    const std::vector<int32_t> subgraphOutputs({1});
    flatbuffers::Offset<SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&ceilOperator, 1));

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

template <typename T>
void DelegateOptionTest(tflite::TensorType tensorType,
                        std::vector<int32_t>& tensorShape,
                        std::vector<T>& input0Values,
                        std::vector<T>& input1Values,
                        std::vector<T>& input2Values,
                        std::vector<T>& expectedOutputValues,
                        const armnnDelegate::DelegateOptions& delegateOptions,
                        float quantScale = 1.0f,
                        int quantOffset  = 0)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateAddDivTfLiteModel(tensorType,
                                                            tensorShape,
                                                            quantScale,
                                                            quantOffset);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(input0Values, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(input1Values, 1) == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(input2Values, 2) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, delegateOptions);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(input0Values, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(input1Values, 1) == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(input2Values, 2) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, tensorShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

template <typename T>
void DelegateOptionNoFallbackTest(tflite::TensorType tensorType,
                                  std::vector<int32_t>& tensorShape,
                                  std::vector<T>& inputValues,
                                  std::vector<T>& expectedOutputValues,
                                  const armnnDelegate::DelegateOptions& delegateOptions,
                                  float quantScale = 1.0f,
                                  int quantOffset  = 0)
{
    using namespace delegateTestInterpreter;
    std::vector<char> modelBuffer = CreateCosTfLiteModel(tensorType,
                                                         tensorShape,
                                                         quantScale,
                                                         quantOffset);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);
    tfLiteInterpreter.Cleanup();

    try
    {
        auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, delegateOptions);
        CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
        CHECK(armnnInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
        CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
        std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
        std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);
        armnnInterpreter.Cleanup();

        armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
        armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, tensorShape);
    }
    catch (const armnn::Exception& e)
    {
        // Forward the exception message to std::cout
        std::cout << e.what() << std::endl;
    }
}

} // anonymous namespace