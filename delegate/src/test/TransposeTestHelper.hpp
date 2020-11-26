//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

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
std::vector<char> CreateTransposeTfLiteModel(tflite::TensorType tensorType,
                                             const std::vector <int32_t>& input0TensorShape,
                                             const std::vector <int32_t>& inputPermVecShape,
                                             const std::vector <int32_t>& outputTensorShape,
                                             const std::vector<int32_t>& inputPermVec)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;
    std::array<flatbuffers::Offset<tflite::Buffer>, 2> buffers;
    buffers[0] = CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({}));
    buffers[1] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(inputPermVec.data()),
                                                             sizeof(int32_t) * inputPermVec.size()));
    std::array<flatbuffers::Offset<Tensor>, 3> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(input0TensorShape.data(),
                                                                      input0TensorShape.size()),
                              tensorType, 0);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputPermVecShape.data(),
                                                                      inputPermVecShape.size()),
                              tflite::TensorType_INT32, 1,
                              flatBufferBuilder.CreateString("permutation_vector"));
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType);
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
                    flatBufferBuilder.CreateVector(buffers.data(), buffers.size()));
    flatBufferBuilder.Finish(flatbufferModel);
    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

void TransposeFP32Test(std::vector<armnn::BackendId>& backends)
{
    using namespace tflite;

    // set test input data
    std::vector<int32_t> input0Shape {4, 2, 3};
    std::vector<int32_t> inputPermVecShape {3};
    std::vector<int32_t> outputShape {2, 3, 4};

    std::vector<float> input0Values = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    std::vector<int32_t> inputPermVec = {2, 0, 1};
    std::vector<float> expectedOutputValues = {0, 3, 6, 9, 12, 15, 18, 21, 1, 4, 7, 10,
                                               13, 16, 19, 22, 2, 5, 8, 11, 14, 17, 20, 23};

    // create model
    std::vector<char> modelBuffer = CreateTransposeTfLiteModel(::tflite::TensorType_FLOAT32,
                                                               input0Shape,
                                                               inputPermVecShape,
                                                               outputShape,
                                                               inputPermVec);

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

    // Set input data for tflite
    auto tfLiteInterpreterInput0Id = tfLiteInterpreter->inputs()[0];
    auto tfLiteInterpreterInput0Data = tfLiteInterpreter->typed_tensor<float>(tfLiteInterpreterInput0Id);
    for (unsigned int i = 0; i < input0Values.size(); ++i)
    {
        tfLiteInterpreterInput0Data[i] = input0Values[i];
    }

    auto tfLiteInterpreterInput1Id = tfLiteInterpreter->inputs()[1];
    auto tfLiteInterpreterInput1Data = tfLiteInterpreter->typed_tensor<int32_t>(tfLiteInterpreterInput1Id);
    for (unsigned int i = 0; i < inputPermVec.size(); ++i)
    {
        tfLiteInterpreterInput1Data[i] = inputPermVec[i];
    }

    //Set input data for armnn delegate
    auto armnnDelegateInput0Id = armnnDelegateInterpreter->inputs()[0];
    auto armnnDelegateInput0Data = armnnDelegateInterpreter->typed_tensor<float>(armnnDelegateInput0Id);
    for (unsigned int i = 0; i < input0Values.size(); ++i)
    {
        armnnDelegateInput0Data[i] = input0Values[i];
    }

    auto armnnDelegateInput1Id = armnnDelegateInterpreter->inputs()[1];
    auto armnnDelegateInput1Data = armnnDelegateInterpreter->typed_tensor<int32_t>(armnnDelegateInput1Id);
    for (unsigned int i = 0; i < inputPermVec.size(); ++i)
    {
        armnnDelegateInput1Data[i] = inputPermVec[i];
    }

    // Run EnqueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    // Compare output data
    auto tfLiteInterpreterOutputId = tfLiteInterpreter->outputs()[0];
    auto tfLiteInterpreterOutputData = tfLiteInterpreter->typed_tensor<float>(tfLiteInterpreterOutputId);
    auto armnnDelegateOutputId = armnnDelegateInterpreter->outputs()[0];
    auto armnnDelegateOutputData = armnnDelegateInterpreter->typed_tensor<float>(armnnDelegateOutputId);
    for (size_t i = 0; i < expectedOutputValues.size(); ++i)
    {
        CHECK(expectedOutputValues[i] == armnnDelegateOutputData[i]);
        CHECK(tfLiteInterpreterOutputData[i] == expectedOutputValues[i]);
        CHECK(tfLiteInterpreterOutputData[i] == armnnDelegateOutputData[i]);
    }

    armnnDelegateInterpreter.reset(nullptr);
}
}
