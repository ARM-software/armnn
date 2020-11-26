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

std::vector<char> CreateResizeTfLiteModel(tflite::BuiltinOperator operatorCode,
                                          tflite::TensorType inputTensorType,
                                          const std::vector <int32_t>& inputTensorShape,
                                          const std::vector <int32_t>& sizeTensorData,
                                          const std::vector <int32_t>& sizeTensorShape,
                                          const std::vector <int32_t>& outputTensorShape)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));
    buffers.push_back(CreateBuffer(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector(
                                           reinterpret_cast<const uint8_t*>(sizeTensorData.data()),
                                           sizeof(int32_t) * sizeTensorData.size())));

    std::array<flatbuffers::Offset<Tensor>, 3> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(), inputTensorShape.size()),
                              inputTensorType,
                              0,
                              flatBufferBuilder.CreateString("input_tensor"));

    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(sizeTensorShape.data(),
                                                                      sizeTensorShape.size()),
                              TensorType_INT32,
                              1,
                              flatBufferBuilder.CreateString("size_input_tensor"));

    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              inputTensorType,
                              0,
                              flatBufferBuilder.CreateString("output_tensor"));

    // Create Operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_NONE;
    flatbuffers::Offset<void> operatorBuiltinOption = 0;
    switch (operatorCode)
    {
        case BuiltinOperator_RESIZE_BILINEAR:
        {
            operatorBuiltinOption = CreateResizeBilinearOptions(flatBufferBuilder, false, false).Union();
            operatorBuiltinOptionsType = tflite::BuiltinOptions_ResizeBilinearOptions;
            break;
        }
        case BuiltinOperator_RESIZE_NEAREST_NEIGHBOR:
        {
            operatorBuiltinOption = CreateResizeNearestNeighborOptions(flatBufferBuilder, false, false).Union();
            operatorBuiltinOptionsType = tflite::BuiltinOptions_ResizeNearestNeighborOptions;
            break;
        }
        default:
            break;
    }

    const std::vector<int> operatorInputs{0, 1};
    const std::vector<int> operatorOutputs{2};
    flatbuffers::Offset <Operator> resizeOperator =
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
                       flatBufferBuilder.CreateVector(&resizeOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Resize Biliniar Operator Model");
    flatbuffers::Offset <OperatorCode> opCode = CreateOperatorCode(flatBufferBuilder, operatorCode);

    flatbuffers::Offset <Model> flatbufferModel =
        CreateModel(flatBufferBuilder,
                    TFLITE_SCHEMA_VERSION,
                    flatBufferBuilder.CreateVector(&opCode, 1),
                    flatBufferBuilder.CreateVector(&subgraph, 1),
                    modelDescription,
                    flatBufferBuilder.CreateVector(buffers.data(), buffers.size()));

    flatBufferBuilder.Finish(flatbufferModel);

    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

void ResizeFP32TestImpl(tflite::BuiltinOperator operatorCode,
                        std::vector<armnn::BackendId>& backends,
                        std::vector<float>& input1Values,
                        std::vector<int32_t> input1Shape,
                        std::vector<int32_t> input2NewShape,
                        std::vector<int32_t> input2Shape,
                        std::vector<float>& expectedOutputValues,
                        std::vector<int32_t> expectedOutputShape)
{
    using namespace tflite;

    std::vector<char> modelBuffer = CreateResizeTfLiteModel(operatorCode,
                                                            ::tflite::TensorType_FLOAT32,
                                                            input1Shape,
                                                            input2NewShape,
                                                            input2Shape,
                                                            expectedOutputShape);

    const Model* tfLiteModel = GetModel(modelBuffer.data());

    // The model will be executed using tflite and using the armnn delegate so that the outputs
    // can be compared.

    // Create TfLite Interpreter with armnn delegate
    std::unique_ptr<Interpreter> armnnDelegateInterpreter;
    CHECK(InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
              (&armnnDelegateInterpreter) == kTfLiteOk);
    CHECK(armnnDelegateInterpreter != nullptr);
    CHECK(armnnDelegateInterpreter->AllocateTensors() == kTfLiteOk);

    // Create TfLite Interpreter without armnn delegate
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

    // Set input data for the armnn interpreter
    armnnDelegate::FillInput(armnnDelegateInterpreter, 0, input1Values);
    armnnDelegate::FillInput(armnnDelegateInterpreter, 1, input2NewShape);

    // Set input data for the tflite interpreter
    armnnDelegate::FillInput(tfLiteInterpreter, 0, input1Values);
    armnnDelegate::FillInput(tfLiteInterpreter, 1, input2NewShape);

    // Run EnqueWorkload
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);

    // Compare output data
    auto tfLiteDelegateOutputId = tfLiteInterpreter->outputs()[0];
    auto tfLiteDelageOutputData = tfLiteInterpreter->typed_tensor<float>(tfLiteDelegateOutputId);
    auto armnnDelegateOutputId = armnnDelegateInterpreter->outputs()[0];
    auto armnnDelegateOutputData = armnnDelegateInterpreter->typed_tensor<float>(armnnDelegateOutputId);
    for (size_t i = 0; i < expectedOutputValues.size(); i++)
    {
        CHECK(expectedOutputValues[i] == doctest::Approx(armnnDelegateOutputData[i]));
        CHECK(armnnDelegateOutputData[i] == doctest::Approx(tfLiteDelageOutputData[i]));
    }

    armnnDelegateInterpreter.reset(nullptr);
}

} // anonymous namespace