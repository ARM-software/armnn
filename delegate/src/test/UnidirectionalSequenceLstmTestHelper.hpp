//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
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
#include <tensorflow/lite/c/common.h>

#include <doctest/doctest.h>


#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/TypesUtils.hpp>

#include <armnn/Types.hpp>

#include <initializer_list>
#include <iterator>
#include <vector>

namespace
{

template <typename T>
std::vector<char> CreateUnidirectionalSequenceLstmTfLiteModel(tflite::TensorType tensorType,
                                                              int32_t batchSize,
                                                              int32_t timeSize,
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
                                                              const std::vector<float>& inputGateBias,
                                                              const std::vector<float>& forgetGateBias,
                                                              const std::vector<float>& cellBias,
                                                              const std::vector<float>& outputGateBias,
                                                              bool hasProjectionWeights,
                                                              const std::vector<T>& projectionWeights,
                                                              bool hasProjectionBias,
                                                              const std::vector<float>& projectionBias,
                                                              bool hasInputLayerNormWeights,
                                                              const std::vector<float>& inputLayerNormWeights,
                                                              bool hasForgetLayerNormWeights,
                                                              const std::vector<float>& forgetLayerNormWeights,
                                                              bool hasCellLayerNormWeights,
                                                              const std::vector<float>& cellLayerNormWeights,
                                                              bool hasOutputLayerNormWeights,
                                                              const std::vector<float>& outputLayerNormWeights,
                                                              tflite::ActivationFunctionType activationFunction,
                                                              float clippingThresCell,
                                                              float clippingThresProj,
                                                              bool isTimeMajor,
                                                              float quantScale,
                                                              int quantOffset  = 0)
{

    std::vector<int32_t> tensorInfo0{};
    std::vector<int32_t> tensorInfoNumUnits{numUnits};
    std::vector<int32_t> tensorInfoInputSize{numUnits, inputSize};
    std::vector<int32_t> tensorInfoOutputSize{numUnits, outputSize};

    std::vector<int32_t> inputShape;
    std::vector<int32_t> outputShape;
    if (isTimeMajor)
    {
        inputShape  = {timeSize, batchSize, inputSize};
        outputShape = {timeSize, batchSize, outputSize};
    }
    else
    {
        inputShape  = {batchSize, timeSize, inputSize};
        outputShape = {batchSize, timeSize, outputSize};
    }
    std::vector<int32_t> outputStateInDimensions{batchSize, outputSize};
    std::vector<int32_t> cellStateInDimensions{batchSize, numUnits};
    std::vector<int32_t> projectionWeightDimensions{outputSize, numUnits};
    std::vector<int32_t> projectionBiasDimensions{outputSize};

    std::vector<int> operatorInputs;
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    std::vector<flatbuffers::Offset<Tensor>> tensors;

    auto quantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({ 1.0f }),
                                     flatBufferBuilder.CreateVector<int64_t>({ 0 }));

    auto weightQuantizationParameters =
        CreateQuantizationParameters(flatBufferBuilder,
                                     0,
                                     0,
                                     flatBufferBuilder.CreateVector<float>({ quantScale }),
                                     flatBufferBuilder.CreateVector<int64_t>({ quantOffset }));

    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(inputShape.data(),
                                                                           inputShape.size()),
                                   ::tflite::TensorType_FLOAT32,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("input_0")));
    operatorInputs.push_back(buffers.size() - 1);

    if (hasInputToInputWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(inputToInputWeights.data()),
                                                        sizeof(T) * inputToInputWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoInputSize.data(),
                                                                               tensorInfoInputSize.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("inputToInputWeights"),
                                       weightQuantizationParameters));
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
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoInputSize.data(),
                                                                           tensorInfoInputSize.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("inputToForgetWeights"),
                                   weightQuantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(inputToCellWeights.data()),
                                                    sizeof(T) * inputToCellWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoInputSize.data(),
                                                                           tensorInfoInputSize.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("inputToCellWeights"),
                                   weightQuantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(inputToOutputWeights.data()),
                                                    sizeof(T) * inputToOutputWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoInputSize.data(),
                                                                           tensorInfoInputSize.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("inputToOutputWeights"),
                                   weightQuantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    if (hasRecurrentToInputWeights)
    {
        buffers.push_back(CreateBuffer(
            flatBufferBuilder,
            flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(recurrentToInputWeights.data()),
                                           sizeof(T) * recurrentToInputWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoOutputSize.data(),
                                                                               tensorInfoOutputSize.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("recurrentToInputWeights"),
                                       weightQuantizationParameters));
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
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoOutputSize.data(),
                                                                           tensorInfoOutputSize.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("recurrentToForgetWeights"),
                                   weightQuantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(recurrentToCellWeights.data()),
                                                    sizeof(T) * recurrentToCellWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoOutputSize.data(),
                                                                           tensorInfoOutputSize.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("recurrentToCellWeights"),
                                   weightQuantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(recurrentToOutputWeights.data()),
                                                    sizeof(T) * recurrentToOutputWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoOutputSize.data(),
                                                                           tensorInfoOutputSize.size()),
                                   tensorType,
                                   buffers.size() - 1 ,
                                   flatBufferBuilder.CreateString("recurrentToOutputWeights"),
                                   weightQuantizationParameters));
    operatorInputs.push_back(buffers.size() - 1);

    if (hasCellToInputWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(cellToInputWeights.data()),
                                                        sizeof(T) * cellToInputWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                               tensorInfoNumUnits.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("cellToInputWeights"),
                                       weightQuantizationParameters));
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
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                               tensorInfoNumUnits.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("cellToForgetWeights"),
                                       weightQuantizationParameters));
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
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                               tensorInfoNumUnits.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("cellToOutputWeights"),
                                       weightQuantizationParameters));
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
                                                        sizeof(float) * inputGateBias.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                               tensorInfoNumUnits.size()),
                                       ::tflite::TensorType_FLOAT32,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("inputGateBias")));
        operatorInputs.push_back(buffers.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(forgetGateBias.data()),
                                                    sizeof(float) * forgetGateBias.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                           tensorInfoNumUnits.size()),
                                   ::tflite::TensorType_FLOAT32,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("forgetGateBias")));
    operatorInputs.push_back(buffers.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(cellBias.data()),
                                                    sizeof(float) * cellBias.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                           tensorInfoNumUnits.size()),
                                   ::tflite::TensorType_FLOAT32,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("cellBias")));
    operatorInputs.push_back(buffers.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(outputGateBias.data()),
                                                    sizeof(float) * outputGateBias.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                           tensorInfoNumUnits.size()),
                                   ::tflite::TensorType_FLOAT32,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("outputGateBias")));
    operatorInputs.push_back(buffers.size() - 1);

    if (hasProjectionWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t *>(projectionWeights.data()),
                                                        sizeof(T) * projectionWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(projectionWeightDimensions.data(),
                                                                               projectionWeightDimensions.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("projectionWeights"),
                                       weightQuantizationParameters));
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
                                                        sizeof(float) * projectionBias.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(projectionBiasDimensions.data(),
                                                                               projectionBiasDimensions.size()),
                                       ::tflite::TensorType_FLOAT32,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("projectionBias")));
        operatorInputs.push_back(buffers.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(outputStateInDimensions.data(),
                                                                           outputStateInDimensions.size()),
                                   ::tflite::TensorType_FLOAT32,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("outputStateInInfo"),
                                   quantizationParameters,
                                   true));
    operatorInputs.push_back(buffers.size() - 1);

    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(cellStateInDimensions.data(),
                                                                           cellStateInDimensions.size()),
                                   ::tflite::TensorType_FLOAT32,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("cellStateInInfo"),
                                   quantizationParameters,
                                   true));
    operatorInputs.push_back(buffers.size() - 1);

    if (hasInputLayerNormWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(
                                              reinterpret_cast<const uint8_t *>(inputLayerNormWeights.data()),
                                              sizeof(float) * inputLayerNormWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                               tensorInfoNumUnits.size()),
                                       ::tflite::TensorType_FLOAT32,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("inputLayerNormWeights")));
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
                                              sizeof(float) * forgetLayerNormWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                               tensorInfoNumUnits.size()),
                                       ::tflite::TensorType_FLOAT32,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("forgetLayerNormWeights")));
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
                                                        sizeof(float) * cellLayerNormWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                               tensorInfoNumUnits.size()),
                                       ::tflite::TensorType_FLOAT32,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("cellLayerNormWeights")));
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
                             sizeof(float) * outputLayerNormWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                               tensorInfoNumUnits.size()),
                                       ::tflite::TensorType_FLOAT32,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("outputLayerNormWeights")));
        operatorInputs.push_back(buffers.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }
    int outputBufferId = buffers.size();
    buffers.push_back(CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({})));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(outputShape.data(),
                                                                           outputShape.size()),
                                   ::tflite::TensorType_FLOAT32,
                                   outputBufferId,
                                   flatBufferBuilder.CreateString("output")));
    std::vector<int> operatorOutputs;
    operatorOutputs.push_back(buffers.size() - 1);

    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = BuiltinOptions_UnidirectionalSequenceLSTMOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions =
        CreateUnidirectionalSequenceLSTMOptions(flatBufferBuilder,
                          activationFunction,
                          clippingThresCell,
                          clippingThresProj,
                          isTimeMajor).Union();

    flatbuffers::Offset<Operator> lstmOperator =
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
        flatBufferBuilder.CreateString("ArmnnDelegate: UnidirectionalSequenceLSTM Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode =
        CreateOperatorCode(flatBufferBuilder, tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM);

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
void UnidirectionalSequenceLstmTestImpl(std::vector<armnn::BackendId>& backends,
                                        tflite::TensorType tensorType,
                                        int32_t batchSize,
                                        int32_t timeSize,
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
                                        const std::vector<float>& inputGateBias,
                                        const std::vector<float>& forgetGateBias,
                                        const std::vector<float>& cellBias,
                                        const std::vector<float>& outputGateBias,
                                        bool hasProjectionWeights,
                                        const std::vector<T>& projectionWeights,
                                        bool hasProjectionBias,
                                        const std::vector<float>& projectionBias,
                                        bool hasInputLayerNormWeights,
                                        const std::vector<float>& inputLayerNormWeights,
                                        bool hasForgetLayerNormWeights,
                                        const std::vector<float>& forgetLayerNormWeights,
                                        bool hasCellLayerNormWeights,
                                        const std::vector<float>& cellLayerNormWeights,
                                        bool hasOutputLayerNormWeights,
                                        const std::vector<float>& outputLayerNormWeights,
                                        std::vector<float>& inputValues,
                                        std::vector<float>& expectedOutputValues,
                                        tflite::ActivationFunctionType activationFunction,
                                        float clippingThresCell,
                                        float clippingThresProj,
                                        bool isTimeMajor,
                                        float quantScale = 0.1f)
{
    using namespace tflite;

    std::vector<char> modelBuffer = CreateUnidirectionalSequenceLstmTfLiteModel(tensorType,
                                                          batchSize,
                                                          timeSize,
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
                                                          clippingThresProj,
                                                          isTimeMajor,
                                                          quantScale);

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
    auto tfLiteDelageInputData = tfLiteInterpreter->typed_tensor<float>(tfLiteDelegateInputId);
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        tfLiteDelageInputData[i] = inputValues[i];
    }

    auto armnnDelegateInputId = armnnDelegateInterpreter->inputs()[0];
    auto armnnDelegateInputData = armnnDelegateInterpreter->typed_tensor<float>(armnnDelegateInputId);
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        armnnDelegateInputData[i] = inputValues[i];
    }

    // Run EnqueueWorkload
    CHECK(tfLiteInterpreter->Invoke() == kTfLiteOk);
    CHECK(armnnDelegateInterpreter->Invoke() == kTfLiteOk);

    // Compare output data
    auto tfLiteDelegateOutputId = tfLiteInterpreter->outputs()[0];
    auto tfLiteDelagateOutputData = tfLiteInterpreter->typed_tensor<float>(tfLiteDelegateOutputId);
    auto armnnDelegateOutputId = armnnDelegateInterpreter->outputs()[0];
    auto armnnDelegateOutputData = armnnDelegateInterpreter->typed_tensor<float>(armnnDelegateOutputId);

    if (tensorType == ::tflite::TensorType_INT8)
    {
        // Allow 2% tolerance for Quantized weights
        armnnDelegate::CompareData(expectedOutputValues.data(), armnnDelegateOutputData,
                                   expectedOutputValues.size(), 2);
        armnnDelegate::CompareData(expectedOutputValues.data(), tfLiteDelagateOutputData,
                                   expectedOutputValues.size(), 2);
        armnnDelegate::CompareData(tfLiteDelagateOutputData, armnnDelegateOutputData,
                                   expectedOutputValues.size(), 2);
    }
    else
    {
        armnnDelegate::CompareData(expectedOutputValues.data(), armnnDelegateOutputData, expectedOutputValues.size());
        armnnDelegate::CompareData(expectedOutputValues.data(), tfLiteDelagateOutputData, expectedOutputValues.size());
        armnnDelegate::CompareData(tfLiteDelagateOutputData, armnnDelegateOutputData, expectedOutputValues.size());
    }
}

} // anonymous namespace