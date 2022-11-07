//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn_delegate.hpp>

#include "ConvolutionTestHelper.hpp"
#include "TestUtils.hpp"

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

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
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));

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
                              0,
                              flatBufferBuilder.CreateString("input_0"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("input_1"),
                              quantizationParameters);
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("input_2"),
                              quantizationParameters);
    tensors[3] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("add"),
                              quantizationParameters);
    tensors[4] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(tensorShape.data(),
                                                                      tensorShape.size()),
                              tensorType,
                              0,
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

    flatBufferBuilder.Finish(flatbufferModel);

    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

std::vector<char> CreateCeilTfLiteModel(tflite::TensorType tensorType,
                                        const std::vector <int32_t>& tensorShape,
                                        float quantScale = 1.0f,
                                        int quantOffset = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));

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
        CreateOperatorCode(flatBufferBuilder, tflite::BuiltinOperator_CEIL);

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

    flatBufferBuilder.Finish(flatbufferModel);
    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

template <typename T>
void DelegateOptionTest(tflite::TensorType tensorType,
                        const std::vector<armnn::BackendId>& backends,
                        std::vector<int32_t>& tensorShape,
                        std::vector<T>& input0Values,
                        std::vector<T>& input1Values,
                        std::vector<T>& input2Values,
                        std::vector<T>& expectedOutputValues,
                        const armnnDelegate::DelegateOptions& delegateOptions,
                        float quantScale = 1.0f,
                        int quantOffset  = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateAddDivTfLiteModel(tensorType,
                                                            tensorShape,
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
    std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
        theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                         armnnDelegate::TfLiteArmnnDelegateDelete);
    CHECK(theArmnnDelegate != nullptr);
    // Modify armnnDelegateInterpreter to use armnnDelegate
    CHECK(armnnDelegateInterpreter->ModifyGraphWithDelegate(theArmnnDelegate.get()) == kTfLiteOk);

    // Set input data
    armnnDelegate::FillInput(tfLiteInterpreter, 0, input0Values);
    armnnDelegate::FillInput(tfLiteInterpreter, 1, input1Values);
    armnnDelegate::FillInput(tfLiteInterpreter, 2, input2Values);

    armnnDelegate::FillInput(armnnDelegateInterpreter, 0, input0Values);
    armnnDelegate::FillInput(armnnDelegateInterpreter, 1, input1Values);
    armnnDelegate::FillInput(armnnDelegateInterpreter, 2, input2Values);

    // Run EnqueueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    armnnDelegate::CompareOutputData<T>(tfLiteInterpreter, armnnDelegateInterpreter, tensorShape, expectedOutputValues);

    armnnDelegateInterpreter.reset(nullptr);
}

template <typename T>
void DelegateOptionNoFallbackTest(tflite::TensorType tensorType,
                                  const std::vector<armnn::BackendId>& backends,
                                  std::vector<int32_t>& tensorShape,
                                  std::vector<T>& inputValues,
                                  std::vector<T>& expectedOutputValues,
                                  const armnnDelegate::DelegateOptions& delegateOptions,
                                  float quantScale = 1.0f,
                                  int quantOffset  = 0)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateCeilTfLiteModel(tensorType,
                                                          tensorShape,
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
    std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
        theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                         armnnDelegate::TfLiteArmnnDelegateDelete);
    CHECK(theArmnnDelegate != nullptr);
    // Modify armnnDelegateInterpreter to use armnnDelegate
    try
    {
        armnnDelegateInterpreter->ModifyGraphWithDelegate(theArmnnDelegate.get());
    }
    catch (const armnn::Exception& e)
    {
        // Forward the exception message to std::cout
        std::cout << e.what() << std::endl;
    }

    // Set input data
    armnnDelegate::FillInput(tfLiteInterpreter, 0, inputValues);
    armnnDelegate::FillInput(armnnDelegateInterpreter, 0, inputValues);

    // Run EnqueueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    armnnDelegate::CompareOutputData<T>(tfLiteInterpreter, armnnDelegateInterpreter, tensorShape, expectedOutputValues);

    armnnDelegateInterpreter.reset(nullptr);
}

} // anonymous namespace