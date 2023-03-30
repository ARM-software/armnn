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
std::vector<char> CreateLstmTfLiteModel(tflite::TensorType tensorType,
                                        int32_t batchSize,
                                        int32_t inputSize,
                                        int32_t outputSize,
                                        int32_t numUnits,
                                        bool hasInputToInputWeights,
                                        const std::vector<T>& inputToInputWeights,
                                        const std::vector<T>& inputToForgetWeights,
                                        const std::vector<T>& inputToCellWeights,
                                        const std::vector<T>& inputToOutputWeights,
                                        bool hasRecurrentToInputWeights,
                                        const std::vector<T>& recurrentToInputWeights,
                                        const std::vector<T>& recurrentToForgetWeights,
                                        const std::vector<T>& recurrentToCellWeights,
                                        const std::vector<T>& recurrentToOutputWeights,
                                        bool hasCellToInputWeights,
                                        const std::vector<T>& cellToInputWeights,
                                        bool hasCellToForgetWeights,
                                        const std::vector<T>& cellToForgetWeights,
                                        bool hasCellToOutputWeights,
                                        const std::vector<T>& cellToOutputWeights,
                                        bool hasInputGateBias,
                                        const std::vector<T>& inputGateBias,
                                        const std::vector<T>& forgetGateBias,
                                        const std::vector<T>& cellBias,
                                        const std::vector<T>& outputGateBias,
                                        bool hasProjectionWeights,
                                        const std::vector<T>& projectionWeights,
                                        bool hasProjectionBias,
                                        const std::vector<T>& projectionBias,
                                        bool hasInputLayerNormWeights,
                                        const std::vector<T>& inputLayerNormWeights,
                                        bool hasForgetLayerNormWeights,
                                        const std::vector<T>& forgetLayerNormWeights,
                                        bool hasCellLayerNormWeights,
                                        const std::vector<T>& cellLayerNormWeights,
                                        bool hasOutputLayerNormWeights,
                                        const std::vector<T>& outputLayerNormWeights,
                                        tflite::ActivationFunctionType activationFunction,
                                        float clippingThresCell,
                                        float clippingThresProj,
                                        float quantScale = 1.0f,
                                        int quantOffset  = 0,
                                        float outputQuantScale = 2.0f,
                                        int outputQuantOffset  = 0)
{

    std::vector <int32_t> tensorInfo0 {};
    std::vector <int32_t> tensorInfo4 {numUnits};
    std::vector <int32_t> tensorInfo8 {numUnits, static_cast<int32_t>(2)};
    std::vector <int32_t> tensorInfo16 {numUnits, static_cast<int32_t>(4)};

    std::vector<int32_t> inputShape {batchSize , inputSize};
    std::vector<int32_t> outputShape {batchSize , outputSize};

    std::vector<int32_t> outputStateInDimensions{batchSize, outputSize};
    std::vector<int32_t> cellStateInDimensions{batchSize, numUnits};

    std::vector<int> operatorInputs;
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    std::vector<flatbuffers::Offset<Tensor>> tensors;

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

    buffers.push_back(CreateBuffer(flatBufferBuilder));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(inputShape.data(),
                                                                           inputShape.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("input_0"),
                                   quantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    if (hasInputToInputWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(inputToInputWeights.data()),
                                                        sizeof(T) * inputToInputWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfo8.data(),
                                                                               tensorInfo8.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("inputToInputWeights"),
                                       outputQuantizationParameters));
        operatorInputs.push_back(buffers.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(inputToForgetWeights.data()),
                                                    sizeof(T) * inputToForgetWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfo8.data(),
                                                                           tensorInfo8.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("inputToForgetWeights"),
                                   outputQuantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(inputToCellWeights.data()),
                                                    sizeof(T) * inputToCellWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfo8.data(),
                                                                           tensorInfo8.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("inputToCellWeights"),
                                   outputQuantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(inputToOutputWeights.data()),
                                                    sizeof(T) * inputToOutputWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfo8.data(),
                                                                           tensorInfo8.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("inputToOutputWeights"),
                                   outputQuantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    if (hasRecurrentToInputWeights)
    {
        buffers.push_back(CreateBuffer(
            flatBufferBuilder,
            flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(recurrentToInputWeights.data()),
                                           sizeof(T) * recurrentToInputWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfo16.data(),
                                                                               tensorInfo16.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("recurrentToInputWeights"),
                                       outputQuantizationParameters));
        operatorInputs.push_back(buffers.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(recurrentToForgetWeights.data()),
                                                    sizeof(T) * recurrentToForgetWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfo16.data(),
                                                                           tensorInfo16.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("recurrentToForgetWeights"),
                                   outputQuantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(recurrentToCellWeights.data()),
                                                    sizeof(T) * recurrentToCellWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfo16.data(),
                                                                           tensorInfo16.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("recurrentToCellWeights"),
                                   outputQuantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(recurrentToOutputWeights.data()),
                                                    sizeof(T) * recurrentToOutputWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfo16.data(),
                                                                           tensorInfo16.size()),
                                   tensorType,
                                   buffers.size() - 1 ,
                                   flatBufferBuilder.CreateString("recurrentToOutputWeights"),
                                   outputQuantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    if (hasCellToInputWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(cellToInputWeights.data()),
                                                        sizeof(T) * cellToInputWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfo4.data(),
                                                                               tensorInfo4.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("cellToInputWeights"),
                                       outputQuantizationParameters));
        operatorInputs.push_back(buffers.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    if (hasCellToForgetWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(cellToForgetWeights.data()),
                                                        sizeof(T) * cellToForgetWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfo4.data(),
                                                                               tensorInfo4.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("cellToForgetWeights"),
                                       outputQuantizationParameters));
        operatorInputs.push_back(buffers.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    if (hasCellToOutputWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(cellToOutputWeights.data()),
                                                        sizeof(T) * cellToOutputWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfo4.data(),
                                                                               tensorInfo4.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("cellToOutputWeights"),
                                       outputQuantizationParameters));
        operatorInputs.push_back(buffers.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    if (hasInputGateBias)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(inputGateBias.data()),
                                                        sizeof(T) * inputGateBias.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfo4.data(),
                                                                               tensorInfo4.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("inputGateBias"),
                                       outputQuantizationParameters));
        operatorInputs.push_back(buffers.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(forgetGateBias.data()),
                                                    sizeof(T) * forgetGateBias.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfo4.data(),
                                                                           tensorInfo4.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("forgetGateBias"),
                                   outputQuantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(cellBias.data()),
                                                    sizeof(T) * cellBias.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfo4.data(),
                                                                           tensorInfo4.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("cellBias"),
                                   outputQuantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(outputGateBias.data()),
                                                    sizeof(T) * outputGateBias.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfo4.data(),
                                                                           tensorInfo4.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("outputGateBias"),
                                   outputQuantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    if (hasProjectionWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(projectionWeights.data()),
                                                        sizeof(T) * projectionWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfo4.data(),
                                                                               tensorInfo4.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("outputGateBias"),
                                       outputQuantizationParameters));
        operatorInputs.push_back(buffers.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    if (hasProjectionBias)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(projectionBias.data()),
                                                        sizeof(T) * projectionBias.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfo4.data(),
                                                                               tensorInfo4.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("projectionBias"),
                                       outputQuantizationParameters));
        operatorInputs.push_back(buffers.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    buffers.push_back(CreateBuffer(flatBufferBuilder));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(outputStateInDimensions.data(),
                                                                           outputStateInDimensions.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("outputStateInInfo"),
                                   outputQuantizationParameters,
                                   true));
    operatorInputs.push_back(buffers.size() - 1);

    buffers.push_back(CreateBuffer(flatBufferBuilder));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(cellStateInDimensions.data(),
                                                                           cellStateInDimensions.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("cellStateInInfo"),
                                   outputQuantizationParameters,
                                   true));
    operatorInputs.push_back(buffers.size() - 1);

    if (hasInputLayerNormWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(
                                              reinterpret_cast<const uint8_t *>(inputLayerNormWeights.data()),
                                              sizeof(T) * inputLayerNormWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfo4.data(),
                                                                               tensorInfo4.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("inputLayerNormWeights"),
                                       outputQuantizationParameters));
        operatorInputs.push_back(buffers.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    if (hasForgetLayerNormWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(
                                              reinterpret_cast<const uint8_t *>(forgetLayerNormWeights.data()),
                                              sizeof(T) * forgetLayerNormWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfo4.data(),
                                                                               tensorInfo4.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("forgetLayerNormWeights"),
                                       outputQuantizationParameters));
        operatorInputs.push_back(buffers.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    if (hasCellLayerNormWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(cellLayerNormWeights.data()),
                                                        sizeof(T) * cellLayerNormWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfo4.data(),
                                                                               tensorInfo4.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("cellLayerNormWeights"),
                                       outputQuantizationParameters));
        operatorInputs.push_back(buffers.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    if (hasOutputLayerNormWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(
                             reinterpret_cast<const uint8_t *>(outputLayerNormWeights.data()),
                             sizeof(T) * outputLayerNormWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfo4.data(),
                                                                               tensorInfo4.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("outputLayerNormWeights"),
                                       outputQuantizationParameters));
        operatorInputs.push_back(buffers.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }
    int outputBufferId = buffers.size();
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(outputShape.data(),
                                                                           outputShape.size()),
                                   tensorType,
                                   outputBufferId,
                                   flatBufferBuilder.CreateString("output"),
                                   outputQuantizationParameters));
    std::vector<int> operatorOutputs;
    operatorOutputs.push_back(buffers.size() - 1);

    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_LSTMOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions =
        CreateLSTMOptions(flatBufferBuilder,
                          activationFunction,
                          clippingThresCell,
                          clippingThresProj).Union();

    flatbuffers::Offset <Operator> lstmOperator =
        CreateOperator(flatBufferBuilder,
                       0,
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       operatorBuiltinOptionsType, operatorBuiltinOptions);

    flatbuffers::Offset <SubGraph> subgraph =
        CreateSubGraph(flatBufferBuilder,
                       flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                       flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                       flatBufferBuilder.CreateVector(&lstmOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
        flatBufferBuilder.CreateString("ArmnnDelegate: LSTM Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         tflite::BuiltinOperator_LSTM);

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
void LstmTestImpl(std::vector<armnn::BackendId>& backends,
                  tflite::TensorType tensorType,
                  int32_t batchSize,
                  int32_t inputSize,
                  int32_t outputSize,
                  int32_t numUnits,
                  bool hasInputToInputWeights,
                  const std::vector<T>& inputToInputWeights,
                  const std::vector<T>& inputToForgetWeights,
                  const std::vector<T>& inputToCellWeights,
                  const std::vector<T>& inputToOutputWeights,
                  bool hasRecurrentToInputWeights,
                  const std::vector<T>& recurrentToInputWeights,
                  const std::vector<T>& recurrentToForgetWeights,
                  const std::vector<T>& recurrentToCellWeights,
                  const std::vector<T>& recurrentToOutputWeights,
                  bool hasCellToInputWeights,
                  const std::vector<T>& cellToInputWeights,
                  bool hasCellToForgetWeights,
                  const std::vector<T>& cellToForgetWeights,
                  bool hasCellToOutputWeights,
                  const std::vector<T>& cellToOutputWeights,
                  bool hasInputGateBias,
                  const std::vector<T>& inputGateBias,
                  const std::vector<T>& forgetGateBias,
                  const std::vector<T>& cellBias,
                  const std::vector<T>& outputGateBias,
                  bool hasProjectionWeights,
                  const std::vector<T>& projectionWeights,
                  bool hasProjectionBias,
                  const std::vector<T>& projectionBias,
                  bool hasInputLayerNormWeights,
                  const std::vector<T>& inputLayerNormWeights,
                  bool hasForgetLayerNormWeights,
                  const std::vector<T>& forgetLayerNormWeights,
                  bool hasCellLayerNormWeights,
                  const std::vector<T>& cellLayerNormWeights,
                  bool hasOutputLayerNormWeights,
                  const std::vector<T>& outputLayerNormWeights,
                  std::vector<T>& inputValues,
                  std::vector<T>& expectedOutputValues,
                  tflite::ActivationFunctionType activationFunction,
                  float clippingThresCell,
                  float clippingThresProj)
{
    using namespace delegateTestInterpreter;

    std::vector<char> modelBuffer = CreateLstmTfLiteModel(tensorType,
                                                          batchSize,
                                                          inputSize,
                                                          outputSize,
                                                          numUnits,
                                                          hasInputToInputWeights,
                                                          inputToInputWeights,
                                                          inputToForgetWeights,
                                                          inputToCellWeights,
                                                          inputToOutputWeights,
                                                          hasRecurrentToInputWeights,
                                                          recurrentToInputWeights,
                                                          recurrentToForgetWeights,
                                                          recurrentToCellWeights,
                                                          recurrentToOutputWeights,
                                                          hasCellToInputWeights,
                                                          cellToInputWeights,
                                                          hasCellToForgetWeights,
                                                          cellToForgetWeights,
                                                          hasCellToOutputWeights,
                                                          cellToOutputWeights,
                                                          hasInputGateBias,
                                                          inputGateBias,
                                                          forgetGateBias,
                                                          cellBias,
                                                          outputGateBias,
                                                          hasProjectionWeights,
                                                          projectionWeights,
                                                          hasProjectionBias,
                                                          projectionBias,
                                                          hasInputLayerNormWeights,
                                                          inputLayerNormWeights,
                                                          hasForgetLayerNormWeights,
                                                          forgetLayerNormWeights,
                                                          hasCellLayerNormWeights,
                                                          cellLayerNormWeights,
                                                          hasOutputLayerNormWeights,
                                                          outputLayerNormWeights,
                                                          activationFunction,
                                                          clippingThresCell,
                                                          clippingThresProj);

    std::vector<int32_t> expectedOutputShape {batchSize , outputSize};

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
    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, expectedOutputShape);

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

} // anonymous namespace