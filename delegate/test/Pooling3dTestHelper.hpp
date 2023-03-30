//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn_delegate.hpp>
#include <DelegateTestInterpreter.hpp>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/flexbuffers.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/kernels/custom_ops_register.h>
#include <tensorflow/lite/version.h>

#include <schema_generated.h>

#include <doctest/doctest.h>

namespace
{
#if defined(ARMNN_POST_TFLITE_2_5)

std::vector<uint8_t> CreateCustomOptions(int, int, int, int, int, int, TfLitePadding);

std::vector<char> CreatePooling3dTfLiteModel(
    std::string poolType,
    tflite::TensorType tensorType,
    const std::vector<int32_t>& inputTensorShape,
    const std::vector<int32_t>& outputTensorShape,
    TfLitePadding padding = kTfLitePaddingSame,
    int32_t strideWidth = 0,
    int32_t strideHeight = 0,
    int32_t strideDepth = 0,
    int32_t filterWidth = 0,
    int32_t filterHeight = 0,
    int32_t filterDepth = 0,
    tflite::ActivationFunctionType fusedActivation = tflite::ActivationFunctionType_NONE,
    float quantScale = 1.0f,
    int quantOffset = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));


    auto quantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({ quantScale }),
                                     flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));

    // Create the input and output tensors
    std::array<flatbuffers::Offset<Tensor>, 2> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);

    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("output"),
                              quantizationParameters);

    // Create the custom options from the function below
    std::vector<uint8_t> customOperatorOptions = CreateCustomOptions(strideHeight, strideWidth, strideDepth,
                                                                     filterHeight, filterWidth, filterDepth, padding);
    // opCodeIndex is created as a uint8_t to avoid map lookup
    uint8_t opCodeIndex = 0;
    // Set the operator name based on the PoolType passed in from the test case
    std::string opName = "";
    if (poolType == "kMax")
    {
        opName = "MaxPool3D";
    }
    else
    {
        opName = "AveragePool3D";
    }
    // To create a custom operator code you pass in the builtin code for custom operators and the name of the custom op
    flatbuffers::Offset<OperatorCode> operatorCode = CreateOperatorCodeDirect(flatBufferBuilder,
                                                                              tflite::BuiltinOperator_CUSTOM,
                                                                              opName.c_str());

    // Create the Operator using the opCodeIndex and custom options. Also sets builtin options to none.
    const std::vector<int32_t> operatorInputs{ 0 };
    const std::vector<int32_t> operatorOutputs{ 1 };
    flatbuffers::Offset<Operator> poolingOperator =
        CreateOperator(flatBufferBuilder,
                       opCodeIndex,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       tflite::BuiltinOptions_NONE,
                       0,
                       flatBufferBuilder.CreateVector<uint8_t>(customOperatorOptions),
                       tflite::CustomOptionsFormat_FLEXBUFFERS);

    // Create the subgraph using the operator created above.
    const std::vector<int> subgraphInputs{ 0 };
    const std::vector<int> subgraphOutputs{ 1 };
    flatbuffers::Offset<SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&poolingOperator, 1));

    flatbuffers::Offset<flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Pooling3d Operator Model");

    // Create the model using operatorCode and the subgraph.
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

template<typename T>
void Pooling3dTest(std::string poolType,
                   tflite::TensorType tensorType,
                   std::vector<armnn::BackendId>& backends,
                   std::vector<int32_t>& inputShape,
                   std::vector<int32_t>& outputShape,
                   std::vector<T>& inputValues,
                   std::vector<T>& expectedOutputValues,
                   TfLitePadding padding = kTfLitePaddingSame,
                   int32_t strideWidth = 0,
                   int32_t strideHeight = 0,
                   int32_t strideDepth = 0,
                   int32_t filterWidth = 0,
                   int32_t filterHeight = 0,
                   int32_t filterDepth = 0,
                   tflite::ActivationFunctionType fusedActivation = tflite::ActivationFunctionType_NONE,
                   float quantScale = 1.0f,
                   int quantOffset = 0)
{
    using namespace delegateTestInterpreter;
    // Create the single op model buffer
    std::vector<char> modelBuffer = CreatePooling3dTfLiteModel(poolType,
                                                               tensorType,
                                                               inputShape,
                                                               outputShape,
                                                               padding,
                                                               strideWidth,
                                                               strideHeight,
                                                               strideDepth,
                                                               filterWidth,
                                                               filterHeight,
                                                               filterDepth,
                                                               fusedActivation,
                                                               quantScale,
                                                               quantOffset);

    std::string opType = "";
    if (poolType == "kMax")
    {
        opType = "MaxPool3D";
    }
    else
    {
        opType = "AveragePool3D";
    }

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer, opType);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends, opType);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, outputShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

// Function to create the flexbuffer custom options for the custom pooling3d operator.
std::vector<uint8_t> CreateCustomOptions(int strideHeight, int strideWidth, int strideDepth,
                                         int filterHeight, int filterWidth, int filterDepth, TfLitePadding padding)
{
    auto flex_builder = std::make_unique<flexbuffers::Builder>();
    size_t map_start = flex_builder->StartMap();
    flex_builder->String("data_format", "NDHWC");
    // Padding is created as a key and padding type. Only VALID and SAME supported
    if (padding == kTfLitePaddingValid)
    {
        flex_builder->String("padding", "VALID");
    }
    else
    {
        flex_builder->String("padding", "SAME");
    }

    // Vector of filter dimensions in order ( 1, Depth, Height, Width, 1 )
    auto start = flex_builder->StartVector("ksize");
    flex_builder->Add(1);
    flex_builder->Add(filterDepth);
    flex_builder->Add(filterHeight);
    flex_builder->Add(filterWidth);
    flex_builder->Add(1);
    // EndVector( start, bool typed, bool fixed)
    flex_builder->EndVector(start, true, false);

    // Vector of stride dimensions in order ( 1, Depth, Height, Width, 1 )
    auto stridesStart = flex_builder->StartVector("strides");
    flex_builder->Add(1);
    flex_builder->Add(strideDepth);
    flex_builder->Add(strideHeight);
    flex_builder->Add(strideWidth);
    flex_builder->Add(1);
    // EndVector( stridesStart, bool typed, bool fixed)
    flex_builder->EndVector(stridesStart, true, false);

    flex_builder->EndMap(map_start);
    flex_builder->Finish();

    return flex_builder->GetBuffer();
}
#endif
} // anonymous namespace




