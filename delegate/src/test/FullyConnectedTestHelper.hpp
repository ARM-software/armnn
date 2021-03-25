//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

namespace
{

template <typename T>
std::vector<char> CreateFullyConnectedTfLiteModel(tflite::TensorType tensorType,
                                                  tflite::ActivationFunctionType activationType,
                                                  const std::vector <int32_t>& inputTensorShape,
                                                  const std::vector <int32_t>& weightsTensorShape,
                                                  const std::vector <int32_t>& biasTensorShape,
                                                  std::vector <int32_t>& outputTensorShape,
                                                  std::vector <T>& weightsData,
                                                  bool constantWeights = true,
                                                  float quantScale = 1.0f,
                                                  int quantOffset  = 0,
                                                  float outputQuantScale = 2.0f,
                                                  int outputQuantOffset  = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;
    std::array<flatbuffers::Offset<tflite::Buffer>, 3> buffers;
    buffers[0] = CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({}));

    auto biasTensorType = ::tflite::TensorType_FLOAT32;
    if (tensorType == ::tflite::TensorType_INT8)
    {
        biasTensorType = ::tflite::TensorType_INT32;
    }
    if (constantWeights)
    {
        buffers[1] = CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(weightsData.data()),
                                                    sizeof(T) * weightsData.size()));

        if (tensorType == ::tflite::TensorType_INT8)
        {
            std::vector<int32_t> biasData = { 10 };
            buffers[2] = CreateBuffer(flatBufferBuilder,
                                      flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(biasData.data()),
                                                                     sizeof(int32_t) * biasData.size()));

        }
        else
        {
            std::vector<float> biasData = { 10 };
            buffers[2] = CreateBuffer(flatBufferBuilder,
                                      flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(biasData.data()),
                                                                     sizeof(float) * biasData.size()));
        }
    }
    else
    {
        buffers[1] = CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({}));
        buffers[2] = CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({}));
    }

    auto quantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({ quantScale }),
                                     flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));

    auto outputQuantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({ outputQuantScale }),
                                     flatBufferBuilder.CreateVector<int64_t>({ outputQuantOffset }));

    std::array<flatbuffers::Offset<Tensor>, 4> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("input_0"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(weightsTensorShape.data(),
                                                                      weightsTensorShape.size()),
                              tensorType,
                              1,
                              flatBufferBuilder.CreateString("weights"),
                              quantizationParameters);
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(biasTensorShape.data(),
                                                                      biasTensorShape.size()),
                              biasTensorType,
                              2,
                              flatBufferBuilder.CreateString("bias"),
                              quantizationParameters);

    tensors[3] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("output"),
                              outputQuantizationParameters);


    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_FullyConnectedOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions =
        CreateFullyConnectedOptions(flatBufferBuilder,
                                    activationType,
                                    FullyConnectedOptionsWeightsFormat_DEFAULT, false).Union();

    const std::vector<int> operatorInputs{0, 1, 2};
    const std::vector<int> operatorOutputs{3};
    flatbuffers::Offset <Operator> fullyConnectedOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       operatorBuiltinOptionsType, operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{0, 1, 2};
    const std::vector<int> subgraphOutputs{3};
    flatbuffers::Offset <SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&fullyConnectedOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: FullyConnected Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         tflite::BuiltinOperator_FULLY_CONNECTED);

    flatbuffers::Offset <Model> flatbufferModel =
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
void FullyConnectedTest(std::vector<armnn::BackendId>& backends,
                        tflite::TensorType tensorType,
                        tflite::ActivationFunctionType activationType,
                        const std::vector <int32_t>& inputTensorShape,
                        const std::vector <int32_t>& weightsTensorShape,
                        const std::vector <int32_t>& biasTensorShape,
                        std::vector <int32_t>& outputTensorShape,
                        std::vector <T>& inputValues,
                        std::vector <T>& expectedOutputValues,
                        std::vector <T>& weightsData,
                        bool constantWeights = true,
                        float quantScale = 1.0f,
                        int quantOffset  = 0)
{
    using namespace tflite;

    std::vector<char> modelBuffer = CreateFullyConnectedTfLiteModel(tensorType,
                                                                    activationType,
                                                                    inputTensorShape,
                                                                    weightsTensorShape,
                                                                    biasTensorShape,
                                                                    outputTensorShape,
                                                                    weightsData,
                                                                    constantWeights,
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
    armnnDelegate::DelegateOptions delegateOptions(backends);
    std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
    theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                     armnnDelegate::TfLiteArmnnDelegateDelete);
    CHECK(theArmnnDelegate != nullptr);

    // Modify armnnDelegateInterpreter to use armnnDelegate
    CHECK(armnnDelegateInterpreter->ModifyGraphWithDelegate(theArmnnDelegate.get()) == kTfLiteOk);

    // Set input data
    armnnDelegate::FillInput<T>(tfLiteInterpreter, 0, inputValues);
    armnnDelegate::FillInput<T>(armnnDelegateInterpreter, 0, inputValues);

    if (!constantWeights)
    {
        armnnDelegate::FillInput<T>(tfLiteInterpreter, 1, weightsData);
        armnnDelegate::FillInput<T>(armnnDelegateInterpreter, 1, weightsData);

        if (tensorType == ::tflite::TensorType_INT8)
        {
            std::vector <int32_t> biasData = {10};
            armnnDelegate::FillInput<int32_t>(tfLiteInterpreter, 2, biasData);
            armnnDelegate::FillInput<int32_t>(armnnDelegateInterpreter, 2, biasData);
        }
        else
        {
            std::vector<float> biasData = {10};
            armnnDelegate::FillInput<float>(tfLiteInterpreter, 2, biasData);
            armnnDelegate::FillInput<float>(armnnDelegateInterpreter, 2, biasData);
        }
    }

    // Run EnqueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    // Compare output data
    armnnDelegate::CompareOutputData<T>(tfLiteInterpreter,
                                        armnnDelegateInterpreter,
                                        outputTensorShape,
                                        expectedOutputValues);
    armnnDelegateInterpreter.reset(nullptr);
}

} // anonymous namespace