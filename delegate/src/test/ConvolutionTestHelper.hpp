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

template <typename T, typename B = float>
std::vector<char> CreateConv2dTfLiteModel(tflite::BuiltinOperator convolutionOperatorCode,
                                          tflite::TensorType tensorType,
                                          uint32_t strideX,
                                          uint32_t strideY,
                                          uint32_t dilationX,
                                          uint32_t dilationY,
                                          tflite::Padding padding,
                                          tflite::ActivationFunctionType fused_activation_function,
                                          const std::vector <int32_t>& inputTensorShape,
                                          const std::vector <int32_t>& filterTensorShape,
                                          const std::vector <int32_t>& biasTensorShape,
                                          const std::vector <int32_t>& outputTensorShape,
                                          const std::vector <T>& filterData,
                                          const std::vector <B>& biasData,
                                          float filterScale = 1.0f,
                                          int filterOffset = 0,
                                          float outputQuantScale = 2.0f,
                                          int outputQuantOffset = 0,
                                          float quantScale = 1.0f,
                                          int quantOffset = 0,
                                          int32_t depth_multiplier = 1)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::array<flatbuffers::Offset<tflite::Buffer>, 3> buffers;
    buffers[0] = CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({}));
    buffers[1] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(filterData.data()),
                                                             sizeof(T) * filterData.size()));

    buffers[2] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(biasData.data()),
                                                             sizeof(B) * biasData.size()));

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
    auto filterQuantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({ filterScale }),
                                     flatBufferBuilder.CreateVector<int64_t>({ filterOffset }));

    std::array<flatbuffers::Offset<Tensor>, 4> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(filterTensorShape.data(),
                                                                      filterTensorShape.size()),
                              tensorType,
                              1,
                              flatBufferBuilder.CreateString("filter"),
                              filterQuantizationParameters);

    auto biasTensorType = ::tflite::TensorType_FLOAT32;
    if (tensorType == ::tflite::TensorType_INT8 || tensorType == ::tflite::TensorType_UINT8)
    {
        biasTensorType = ::tflite::TensorType_INT32;
    }
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(biasTensorShape.data(), biasTensorShape.size()),
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

    flatbuffers::Offset<void> operatorBuiltinOptions;
    tflite::BuiltinOptions operatorBuiltinOptionsType;

    if(convolutionOperatorCode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D)
    {
        operatorBuiltinOptionsType = tflite::BuiltinOptions_DepthwiseConv2DOptions;
        operatorBuiltinOptions = CreateDepthwiseConv2DOptions(flatBufferBuilder,
                                                              padding,
                                                              strideX,
                                                              strideY,
                                                              depth_multiplier,
                                                              fused_activation_function,
                                                              dilationX,
                                                              dilationY).Union();
    }
    if(convolutionOperatorCode == tflite::BuiltinOperator_CONV_2D)
    {
        operatorBuiltinOptionsType = tflite::BuiltinOptions_Conv2DOptions;
        operatorBuiltinOptions = CreateConv2DOptions(flatBufferBuilder,
                                                     padding,
                                                     strideX,
                                                     strideY,
                                                     fused_activation_function,
                                                     dilationX,
                                                     dilationY).Union();
    }

    // create operator
    const std::vector<int> operatorInputs{{0, 1, 2}};
    const std::vector<int> operatorOutputs{{3}};
    flatbuffers::Offset <Operator> convolutionOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       operatorBuiltinOptionsType,
                       operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{ {0, 1, 2} };
    const std::vector<int> subgraphOutputs{{3}};
    flatbuffers::Offset <SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&convolutionOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: Convolution2d Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder, convolutionOperatorCode);

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

template <typename T, typename B = float>
void ConvolutionTest(tflite::BuiltinOperator convolutionOperatorCode,
                     tflite::TensorType tensorType,
                     uint32_t strideX,
                     uint32_t strideY,
                     uint32_t dilationX,
                     uint32_t dilationY,
                     tflite::Padding padding,
                     tflite::ActivationFunctionType fused_activation_function,
                     std::vector<armnn::BackendId>& backends,
                     std::vector<int32_t>& inputShape,
                     std::vector<int32_t>& filterShape,
                     std::vector<int32_t>& outputShape,
                     std::vector<T>& inputValues,
                     std::vector<T>& filterValues,
                     std::vector<T>& expectedOutputValues,
                     const std::vector<int32_t>& biasShape = {},
                     const std::vector<B>& biasValues = {},
                     float filterScale = 1.0f,
                     int filterOffset = 0,
                     float outputQuantScale = 2.0f,
                     int outputQuantOffset = 0,
                     float quantScale = 1.0f,
                     int quantOffset = 0,
                     int32_t depth_multiplier = 1)

{
    using namespace tflite;

    std::vector<char> modelBuffer;
    modelBuffer = CreateConv2dTfLiteModel(convolutionOperatorCode,
                                          tensorType,
                                          strideX,
                                          strideY,
                                          dilationX,
                                          dilationY,
                                          padding,
                                          fused_activation_function,
                                          inputShape,
                                          filterShape,
                                          biasShape,
                                          outputShape,
                                          filterValues,
                                          biasValues,
                                          filterScale,
                                          filterOffset,
                                          outputQuantScale,
                                          outputQuantOffset,
                                          quantScale,
                                          quantOffset,
                                          depth_multiplier);


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
    auto tfLiteDelegateInputId = tfLiteInterpreter->inputs()[0];
    auto tfLiteDelageInputData = tfLiteInterpreter->typed_tensor<T>(tfLiteDelegateInputId);
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        tfLiteDelageInputData[i] = inputValues[i];
    }

    auto armnnDelegateInputId = armnnDelegateInterpreter->inputs()[0];
    auto armnnDelegateInputData = armnnDelegateInterpreter->typed_tensor<T>(armnnDelegateInputId);
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        armnnDelegateInputData[i] = inputValues[i];
    }
    // Run EnqueueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    // Compare output data
    auto tfLiteDelegateOutputId = tfLiteInterpreter->outputs()[0];
    auto tfLiteDelagateOutputData = tfLiteInterpreter->typed_tensor<T>(tfLiteDelegateOutputId);
    auto armnnDelegateOutputId = armnnDelegateInterpreter->outputs()[0];
    auto armnnDelegateOutputData = armnnDelegateInterpreter->typed_tensor<T>(armnnDelegateOutputId);
    for (size_t i = 0; i < expectedOutputValues.size(); i++)
    {
        CHECK(tfLiteDelagateOutputData[i] == armnnDelegateOutputData[i]);
        CHECK(doctest::Approx(tfLiteDelagateOutputData[i]).epsilon(0.000001f) == expectedOutputValues[i]);
        CHECK(doctest::Approx(armnnDelegateOutputData[i]).epsilon(0.000001f) == expectedOutputValues[i]);
    }
}

template <typename T>
std::vector<char> CreateTransposeConvTfLiteModel(tflite::TensorType tensorType,
                                                 uint32_t strideX,
                                                 uint32_t strideY,
                                                 tflite::Padding padding,
                                                 const std::vector <int32_t>& transposeTensorShape,
                                                 const std::vector <int32_t>& filterTensorShape,
                                                 const std::vector <int32_t>& inputTensorShape,
                                                 const std::vector <int32_t>& outputTensorShape,
                                                 const std::vector <int32_t>& transposeData,
                                                 const std::vector <T>& filterData,
                                                 float filterScale = 1.0f,
                                                 int filterOffset = 0,
                                                 float outputQuantScale = 2.0f,
                                                 int outputQuantOffset = 0,
                                                 float quantScale = 1.0f,
                                                 int quantOffset = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::array<flatbuffers::Offset<tflite::Buffer>, 3> buffers;
    buffers[0] = CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({}));
    buffers[1] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(transposeData.data()),
                                                             sizeof(int32_t) * transposeData.size()));
    buffers[2] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(filterData.data()),
                                                             sizeof(T) * filterData.size()));

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
    auto filterQuantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({ filterScale }),
                                     flatBufferBuilder.CreateVector<int64_t>({ filterOffset }));

    std::array<flatbuffers::Offset<Tensor>, 4> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(transposeTensorShape.data(),
                              transposeTensorShape.size()),
                              tflite::TensorType_INT32,
                              1);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(filterTensorShape.data(),
                              filterTensorShape.size()),
                              tensorType,
                              2,
                              flatBufferBuilder.CreateString("filter"),
                              filterQuantizationParameters);
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                              inputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);
    tensors[3] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                              outputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("output"),
                              outputQuantizationParameters);

    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_TransposeConvOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions =
        CreateTransposeConvOptions(flatBufferBuilder, padding, strideX, strideY).Union();

    // create operator
    const std::vector<int> operatorInputs{{0, 1, 2}};
    const std::vector<int> operatorOutputs{{3}};
    flatbuffers::Offset <Operator> convolutionOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       operatorBuiltinOptionsType,
                       operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{ {0, 1, 2} };
    const std::vector<int> subgraphOutputs{{3}};
    flatbuffers::Offset <SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                       flatBufferBuilder.CreateVector(&convolutionOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: TransposeConv Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode =
        CreateOperatorCode(flatBufferBuilder, tflite::BuiltinOperator_TRANSPOSE_CONV);

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
void TransposeConvTest(std::vector<armnn::BackendId>& backends,
                       tflite::TensorType tensorType,
                       uint32_t strideX,
                       uint32_t strideY,
                       tflite::Padding padding,
                       const std::vector <int32_t>& transposeTensorShape,
                       const std::vector <int32_t>& filterTensorShape,
                       const std::vector <int32_t>& inputTensorShape,
                       const std::vector <int32_t>& outputTensorShape,
                       const std::vector <int32_t>& transposeData,
                       const std::vector <T>& filterData,
                       std::vector<T>& inputValues,
                       std::vector<T>& expectedOutputValues,
                       float filterScale = 1.0f,
                       int filterOffset = 0,
                       float outputQuantScale = 1.0f,
                       int outputQuantOffset = 0,
                       float quantScale = 1.0f,
                       int quantOffset = 0)
{
    using namespace tflite;

    std::vector<char> modelBuffer;
    modelBuffer = CreateTransposeConvTfLiteModel<T>(tensorType,
                                                    strideX,
                                                    strideY,
                                                    padding,
                                                    transposeTensorShape,
                                                    filterTensorShape,
                                                    inputTensorShape,
                                                    outputTensorShape,
                                                    transposeData,
                                                    filterData,
                                                    filterScale,
                                                    filterOffset,
                                                    outputQuantScale,
                                                    outputQuantOffset,
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
    auto tfLiteDelegateInputId = tfLiteInterpreter->inputs()[2];
    auto tfLiteDelageInputData = tfLiteInterpreter->typed_tensor<T>(tfLiteDelegateInputId);
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        tfLiteDelageInputData[i] = inputValues[i];
    }

    auto armnnDelegateInputId = armnnDelegateInterpreter->inputs()[2];
    auto armnnDelegateInputData = armnnDelegateInterpreter->typed_tensor<T>(armnnDelegateInputId);
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        armnnDelegateInputData[i] = inputValues[i];
    }
    // Run EnqueueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    // Compare output data
    auto tfLiteDelegateOutputId = tfLiteInterpreter->outputs()[0];
    auto tfLiteDelagateOutputData = tfLiteInterpreter->typed_tensor<T>(tfLiteDelegateOutputId);
    auto armnnDelegateOutputId = armnnDelegateInterpreter->outputs()[0];
    auto armnnDelegateOutputData = armnnDelegateInterpreter->typed_tensor<T>(armnnDelegateOutputId);
    for (size_t i = 0; i < expectedOutputValues.size(); i++)
    {
        CHECK(armnnDelegateOutputData[i] == expectedOutputValues[i]);
        CHECK(tfLiteDelagateOutputData[i] == expectedOutputValues[i]);
        CHECK(tfLiteDelagateOutputData[i] == armnnDelegateOutputData[i]);
    }
}

} // anonymous namespace




