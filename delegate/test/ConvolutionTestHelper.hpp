//
// Copyright © 2020, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn_delegate.hpp>
#include <DelegateTestInterpreter.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <schema_generated.h>
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
                                          const std::vector<float> biasScales = {1.0f},
                                          const std::vector<int64_t> biasOffsets = {0},
                                          const std::vector<float> filterScales = {1.0f},
                                          const std::vector<int64_t> filterOffsets = {0},
                                          float outputQuantScale = 2.0f,
                                          int outputQuantOffset = 0,
                                          float quantScale = 1.0f,
                                          int quantOffset = 0,
                                          int32_t depth_multiplier = 1,
                                          int32_t filterQuantizationDim = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::array<flatbuffers::Offset<tflite::Buffer>, 5> buffers;
    buffers[0] = CreateBuffer(flatBufferBuilder);
    buffers[1] = CreateBuffer(flatBufferBuilder);
    buffers[2] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(filterData.data()),
                                                             sizeof(T) * filterData.size()));

    buffers[3] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(biasData.data()),
                                                             sizeof(B) * biasData.size()));
    buffers[4] = CreateBuffer(flatBufferBuilder);

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
                                         flatBufferBuilder.CreateVector<float>(filterScales),
                                         flatBufferBuilder.CreateVector<int64_t>(filterOffsets),
                                         tflite::QuantizationDetails_NONE,
                                         0,
                                         filterQuantizationDim);

    auto biasQuantizationParameters =
            CreateQuantizationParameters(flatBufferBuilder,
                                         0,
                                         0,
                                         flatBufferBuilder.CreateVector<float>(biasScales),
                                         flatBufferBuilder.CreateVector<int64_t>(biasOffsets));

    std::array<flatbuffers::Offset<Tensor>, 4> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              tensorType,
                              1,
                              flatBufferBuilder.CreateString("input"),
                              quantizationParameters);
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(filterTensorShape.data(),
                                                                      filterTensorShape.size()),
                              tensorType,
                              2,
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
                              3,
                              flatBufferBuilder.CreateString("bias"),
                              biasQuantizationParameters);
    tensors[3] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              4,
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
    const std::vector<int> operatorInputs{0, 1, 2};
    const std::vector<int> operatorOutputs{3};
    flatbuffers::Offset <Operator> convolutionOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       operatorBuiltinOptionsType,
                       operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{0, 1, 2};
    const std::vector<int> subgraphOutputs{3};
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

    flatBufferBuilder.Finish(flatbufferModel, armnnDelegate::FILE_IDENTIFIER);

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
                     const std::vector<float> biasScales = {1.0f},
                     const std::vector<int64_t> biasOffsets = {0},
                     const std::vector<float> filterScales = {1.0f},
                     const std::vector<int64_t> filterOffsets = {0},
                     float outputQuantScale = 2.0f,
                     int outputQuantOffset = 0,
                     float quantScale = 1.0f,
                     int quantOffset = 0,
                     int32_t depth_multiplier = 1,
                     int32_t filterQuantizationDim = 3)

{
    using namespace delegateTestInterpreter;

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
                                          biasScales,
                                          biasOffsets,
                                          filterScales,
                                          filterOffsets,
                                          outputQuantScale,
                                          outputQuantOffset,
                                          quantScale,
                                          quantOffset,
                                          depth_multiplier,
                                          filterQuantizationDim);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
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

// Conv3d is only correctly supported for external delegates from TF Lite v2.6, as there was a breaking bug in v2.5.
#if defined(ARMNN_POST_TFLITE_2_5)
template <typename T, typename B = float>
std::vector<char> CreateConv3dTfLiteModel(tflite::BuiltinOperator convolutionOperatorCode,
                                          tflite::TensorType tensorType,
                                          std::vector<uint32_t> strides,
                                          std::vector<uint32_t> dilation,
                                          tflite::Padding padding,
                                          tflite::ActivationFunctionType fused_activation_function,
                                          const std::vector<int32_t>& inputTensorShape,
                                          const std::vector<int32_t>& filterTensorShape,
                                          const std::vector<int32_t>& biasTensorShape,
                                          const std::vector<int32_t>& outputTensorShape,
                                          const std::vector<T>& filterData,
                                          const std::vector<B>& biasData,
                                          const std::vector<float> biasScales = {1.0f},
                                          const std::vector<int64_t> biasOffsets = {0},
                                          const std::vector<float> filterScales = {1.0f},
                                          const std::vector<int64_t> filterOffsets = {0},
                                          float outputQuantScale = 2.0f,
                                          int outputQuantOffset = 0,
                                          float quantScale = 1.0f,
                                          int quantOffset = 0,
                                          int32_t depth_multiplier = 1,
                                          int32_t filterQuantizationDim = 0)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::array<flatbuffers::Offset<tflite::Buffer>, 3> buffers;
    buffers[0] = CreateBuffer(flatBufferBuilder);
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
                                         flatBufferBuilder.CreateVector<float>(filterScales),
                                         flatBufferBuilder.CreateVector<int64_t>(filterOffsets),
                                         tflite::QuantizationDetails_NONE,
                                         0,
                                         filterQuantizationDim);

    auto biasQuantizationParameters =
            CreateQuantizationParameters(flatBufferBuilder,
                                         0,
                                         0,
                                         flatBufferBuilder.CreateVector<float>(biasScales),
                                         flatBufferBuilder.CreateVector<int64_t>(biasOffsets));

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
                              biasQuantizationParameters);
    tensors[3] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("output"),
                              outputQuantizationParameters);

    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_Conv3DOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateConv3DOptions(flatBufferBuilder,
                                                                           padding,
                                                                           strides[2], // Depth
                                                                           strides[0], // Width
                                                                           strides[1], // Height
                                                                           fused_activation_function,
                                                                           dilation[2],
                                                                           dilation[0],
                                                                           dilation[1]).Union();

    // Create operator
    const std::vector<int> operatorInputs{0, 1, 2};
    const std::vector<int> operatorOutputs{3};
    flatbuffers::Offset <Operator> convolutionOperator =
            CreateOperator(flatBufferBuilder,
                           0,
                           flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                           operatorBuiltinOptionsType,
                           operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{0, 1, 2};
    const std::vector<int> subgraphOutputs{3};
    flatbuffers::Offset <SubGraph> subgraph =
            CreateSubGraph(flatBufferBuilder,
                           flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                           flatBufferBuilder.CreateVector(&convolutionOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: Convolution 3d Operator Model");

    // If using an operator with a code greater than 127 then the enum value should be passed as the fifth
    // parameter rather than the second like in other tests.
    flatbuffers::Offset <OperatorCode> operatorCode =
            CreateOperatorCode(flatBufferBuilder, 0, 0, 1, tflite::BuiltinOperator_CONV_3D);

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

template <typename T, typename B = float>
void Convolution3dTest(tflite::BuiltinOperator convolutionOperatorCode,
                       tflite::TensorType tensorType,
                       std::vector<uint32_t> strides,
                       std::vector<uint32_t> dilation,
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
                       const std::vector<float> biasScales = {1.0f},
                       const std::vector<int64_t> biasOffsets = {0},
                       const std::vector<float> filterScales = {1.0f},
                       const std::vector<int64_t> filterOffsets = {0},
                       float outputQuantScale = 2.0f,
                       int outputQuantOffset = 0,
                       float quantScale = 1.0f,
                       int quantOffset = 0,
                       int32_t depth_multiplier = 1,
                       int32_t filterQuantizationDim = 3)
{
    using namespace delegateTestInterpreter;

    std::vector<char> modelBuffer;
    modelBuffer = CreateConv3dTfLiteModel(convolutionOperatorCode,
                                          tensorType,
                                          strides,
                                          dilation,
                                          padding,
                                          fused_activation_function,
                                          inputShape,
                                          filterShape,
                                          biasShape,
                                          outputShape,
                                          filterValues,
                                          biasValues,
                                          biasScales,
                                          biasOffsets,
                                          filterScales,
                                          filterOffsets,
                                          outputQuantScale,
                                          outputQuantOffset,
                                          quantScale,
                                          quantOffset,
                                          depth_multiplier,
                                          filterQuantizationDim);

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(inputValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, outputShape);

    armnnDelegate::CompareData(expectedOutputValues.data(), armnnOutputValues.data(), expectedOutputValues.size(), 1);
    armnnDelegate::CompareData(expectedOutputValues.data(), tfLiteOutputValues.data(), expectedOutputValues.size(), 1);
    armnnDelegate::CompareData(tfLiteOutputValues.data(), armnnOutputValues.data(), expectedOutputValues.size(), 1);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}
#endif

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
    buffers[0] = CreateBuffer(flatBufferBuilder);
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
    const std::vector<int> operatorInputs{0, 1, 2};
    const std::vector<int> operatorOutputs{3};
    flatbuffers::Offset <Operator> convolutionOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       operatorBuiltinOptionsType,
                       operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{0, 1, 2};
    const std::vector<int> subgraphOutputs{3};
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

    flatBufferBuilder.Finish(flatbufferModel, armnnDelegate::FILE_IDENTIFIER);

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
    using namespace delegateTestInterpreter;

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


    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<T>(inputValues, 2) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<T>(inputValues, 2) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<T>       armnnOutputValues = armnnInterpreter.GetOutputResult<T>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputData<T>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, outputTensorShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

} // anonymous namespace




