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

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/TypesUtils.hpp>

#include <armnn/Types.hpp>

#include <initializer_list>
#include <iterator>
#include <vector>

namespace
{

template<typename T>
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
                                                              int quantOffset = 0)
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
    flatbuffers::FlatBufferBuilder                   flatBufferBuilder;
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
    std::vector<flatbuffers::Offset<Tensor>>         tensors;

    auto quantizationParameters =
             CreateQuantizationParameters(flatBufferBuilder,
                                          0,
                                          0,
                                          flatBufferBuilder.CreateVector<float>({1.0f}),
                                          flatBufferBuilder.CreateVector<int64_t>({0}));

    auto weightQuantizationParameters =
             CreateQuantizationParameters(flatBufferBuilder,
                                          0,
                                          0,
                                          flatBufferBuilder.CreateVector<float>({quantScale}),
                                          flatBufferBuilder.CreateVector<int64_t>({quantOffset}));

    buffers.push_back(CreateBuffer(flatBufferBuilder));
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(inputShape.data(),
                                                                           inputShape.size()),
                                   ::tflite::TensorType_FLOAT32,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("input_0")));
    operatorInputs.push_back(tensors.size() - 1);

    if (hasInputToInputWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(
                             reinterpret_cast<const uint8_t*>(inputToInputWeights.data()),
                             sizeof(T) * inputToInputWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoInputSize.data(),
                                                                               tensorInfoInputSize.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("inputToInputWeights"),
                                       weightQuantizationParameters));
        operatorInputs.push_back(tensors.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(
                         reinterpret_cast<const uint8_t*>(inputToForgetWeights.data()),
                         sizeof(T) * inputToForgetWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoInputSize.data(),
                                                                           tensorInfoInputSize.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("inputToForgetWeights"),
                                   weightQuantizationParameters));
    operatorInputs.push_back(tensors.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(
                         reinterpret_cast<const uint8_t*>(inputToCellWeights.data()),
                         sizeof(T) * inputToCellWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoInputSize.data(),
                                                                           tensorInfoInputSize.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("inputToCellWeights"),
                                   weightQuantizationParameters));
    operatorInputs.push_back(tensors.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(
                         reinterpret_cast<const uint8_t*>(inputToOutputWeights.data()),
                         sizeof(T) * inputToOutputWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoInputSize.data(),
                                                                           tensorInfoInputSize.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("inputToOutputWeights"),
                                   weightQuantizationParameters));
    operatorInputs.push_back(tensors.size() - 1);

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
        operatorInputs.push_back(tensors.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(
                                                        recurrentToForgetWeights.data()),
                                                    sizeof(T) * recurrentToForgetWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoOutputSize.data(),
                                                                           tensorInfoOutputSize.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("recurrentToForgetWeights"),
                                   weightQuantizationParameters));
    operatorInputs.push_back(tensors.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(
                                                        recurrentToCellWeights.data()),
                                                    sizeof(T) * recurrentToCellWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoOutputSize.data(),
                                                                           tensorInfoOutputSize.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("recurrentToCellWeights"),
                                   weightQuantizationParameters));
    operatorInputs.push_back(tensors.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(
                                                        recurrentToOutputWeights.data()),
                                                    sizeof(T) * recurrentToOutputWeights.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoOutputSize.data(),
                                                                           tensorInfoOutputSize.size()),
                                   tensorType,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("recurrentToOutputWeights"),
                                   weightQuantizationParameters));
    operatorInputs.push_back(tensors.size() - 1);

    if (hasCellToInputWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(
                                                            cellToInputWeights.data()),
                                                        sizeof(T) * cellToInputWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                               tensorInfoNumUnits.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("cellToInputWeights"),
                                       weightQuantizationParameters));
        operatorInputs.push_back(tensors.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    if (hasCellToForgetWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(
                                                            cellToForgetWeights.data()),
                                                        sizeof(T) * cellToForgetWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                               tensorInfoNumUnits.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("cellToForgetWeights"),
                                       weightQuantizationParameters));
        operatorInputs.push_back(tensors.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    if (hasCellToOutputWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(
                                                            cellToOutputWeights.data()),
                                                        sizeof(T) * cellToOutputWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                               tensorInfoNumUnits.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("cellToOutputWeights"),
                                       weightQuantizationParameters));
        operatorInputs.push_back(tensors.size() - 1);
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
        operatorInputs.push_back(tensors.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(forgetGateBias.data()),
                                                    sizeof(float) * forgetGateBias.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                           tensorInfoNumUnits.size()),
                                   ::tflite::TensorType_FLOAT32,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("forgetGateBias")));
    operatorInputs.push_back(tensors.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(cellBias.data()),
                                                    sizeof(float) * cellBias.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                           tensorInfoNumUnits.size()),
                                   ::tflite::TensorType_FLOAT32,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("cellBias")));
    operatorInputs.push_back(tensors.size() - 1);

    buffers.push_back(
        CreateBuffer(flatBufferBuilder,
                     flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(outputGateBias.data()),
                                                    sizeof(float) * outputGateBias.size())));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                           tensorInfoNumUnits.size()),
                                   ::tflite::TensorType_FLOAT32,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("outputGateBias")));
    operatorInputs.push_back(tensors.size() - 1);

    if (hasProjectionWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(
                             reinterpret_cast<const uint8_t*>(projectionWeights.data()),
                             sizeof(T) * projectionWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(projectionWeightDimensions.data(),
                                                                               projectionWeightDimensions.size()),
                                       tensorType,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("projectionWeights"),
                                       weightQuantizationParameters));
        operatorInputs.push_back(tensors.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    if (hasProjectionBias)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(
                             reinterpret_cast<const uint8_t*>(projectionBias.data()),
                             sizeof(float) * projectionBias.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(projectionBiasDimensions.data(),
                                                                               projectionBiasDimensions.size()),
                                       ::tflite::TensorType_FLOAT32,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("projectionBias")));
        operatorInputs.push_back(tensors.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    buffers.push_back(CreateBuffer(flatBufferBuilder));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(outputStateInDimensions.data(),
                                                                           outputStateInDimensions.size()),
                                   ::tflite::TensorType_FLOAT32,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("outputStateInInfo"),
                                   quantizationParameters,
                                   true));
    operatorInputs.push_back(tensors.size() - 1);

    buffers.push_back(CreateBuffer(flatBufferBuilder));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(cellStateInDimensions.data(),
                                                                           cellStateInDimensions.size()),
                                   ::tflite::TensorType_FLOAT32,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("cellStateInInfo"),
                                   quantizationParameters,
                                   true));
    operatorInputs.push_back(tensors.size() - 1);

    if (hasInputLayerNormWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(
                             reinterpret_cast<const uint8_t*>(inputLayerNormWeights.data()),
                             sizeof(float) * inputLayerNormWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                               tensorInfoNumUnits.size()),
                                       ::tflite::TensorType_FLOAT32,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("inputLayerNormWeights")));
        operatorInputs.push_back(tensors.size() - 1);
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
                             reinterpret_cast<const uint8_t*>(forgetLayerNormWeights.data()),
                             sizeof(float) * forgetLayerNormWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                               tensorInfoNumUnits.size()),
                                       ::tflite::TensorType_FLOAT32,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("forgetLayerNormWeights")));
        operatorInputs.push_back(tensors.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }

    if (hasCellLayerNormWeights)
    {
        buffers.push_back(
            CreateBuffer(flatBufferBuilder,
                         flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(
                                                            cellLayerNormWeights.data()),
                                                        sizeof(float) * cellLayerNormWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                               tensorInfoNumUnits.size()),
                                       ::tflite::TensorType_FLOAT32,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("cellLayerNormWeights")));
        operatorInputs.push_back(tensors.size() - 1);
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
                             reinterpret_cast<const uint8_t*>(outputLayerNormWeights.data()),
                             sizeof(float) * outputLayerNormWeights.size())));
        tensors.push_back(CreateTensor(flatBufferBuilder,
                                       flatBufferBuilder.CreateVector<int32_t>(tensorInfoNumUnits.data(),
                                                                               tensorInfoNumUnits.size()),
                                       ::tflite::TensorType_FLOAT32,
                                       buffers.size() - 1,
                                       flatBufferBuilder.CreateString("outputLayerNormWeights")));
        operatorInputs.push_back(tensors.size() - 1);
    }
    else
    {
        operatorInputs.push_back(kTfLiteOptionalTensor);
    }
    buffers.push_back(CreateBuffer(flatBufferBuilder));
    tensors.push_back(CreateTensor(flatBufferBuilder,
                                   flatBufferBuilder.CreateVector<int32_t>(outputShape.data(),
                                                                           outputShape.size()),
                                   ::tflite::TensorType_FLOAT32,
                                   buffers.size() - 1,
                                   flatBufferBuilder.CreateString("output")));
    std::vector<int> operatorOutputs;
    operatorOutputs.push_back(tensors.size() - 1);

    // create operator
    tflite::BuiltinOptions    operatorBuiltinOptionsType = BuiltinOptions_UnidirectionalSequenceLSTMOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions     =
                                  CreateUnidirectionalSequenceLSTMOptions(flatBufferBuilder,
                                                                          activationFunction,
                                                                          clippingThresCell,
                                                                          clippingThresProj,
                                                                          isTimeMajor).Union();

    flatbuffers::Offset<Operator> lstmOperator =
                                      CreateOperator(flatBufferBuilder,
                                                     0,
                                                     flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(),
                                                                                             operatorInputs.size()),
                                                     flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(),
                                                                                             operatorOutputs.size()),
                                                     operatorBuiltinOptionsType, operatorBuiltinOptions);

    flatbuffers::Offset<SubGraph> subgraph =
                                      CreateSubGraph(flatBufferBuilder,
                                                     flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                                                     flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(),
                                                                                             operatorInputs.size()),
                                                     flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(),
                                                                                             operatorOutputs.size()),
                                                     flatBufferBuilder.CreateVector(&lstmOperator, 1));

    flatbuffers::Offset<flatbuffers::String> modelDescription =
                                                 flatBufferBuilder.CreateString(
                                                     "ArmnnDelegate: UnidirectionalSequenceLSTM Operator Model");
    flatbuffers::Offset<OperatorCode> operatorCode =
                                                 CreateOperatorCode(flatBufferBuilder,
                                                 tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM);

    flatbuffers::Offset<Model> flatbufferModel =
                                   CreateModel(flatBufferBuilder,
                                               TFLITE_SCHEMA_VERSION,
                                               flatBufferBuilder.CreateVector(&operatorCode, 1),
                                               flatBufferBuilder.CreateVector(&subgraph, 1),
                                               modelDescription,
                                               flatBufferBuilder.CreateVector(buffers));

    flatBufferBuilder.Finish(flatbufferModel, armnnDelegate::FILE_IDENTIFIER);

    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

template<typename T>
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
    using namespace delegateTestInterpreter;

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

    std::vector<int32_t> outputShape;
    if (isTimeMajor)
    {
        outputShape = {timeSize, batchSize, outputSize};
    }
    else
    {
        outputShape = {batchSize, timeSize, outputSize};
    }

    // Setup interpreter with just TFLite Runtime.
    auto tfLiteInterpreter = DelegateTestInterpreter(modelBuffer);
    CHECK(tfLiteInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(tfLiteInterpreter.FillInputTensor<float>(inputValues, 0) == kTfLiteOk);
    CHECK(tfLiteInterpreter.Invoke() == kTfLiteOk);
    std::vector<float>   tfLiteOutputValues = tfLiteInterpreter.GetOutputResult<float>(0);
    std::vector<int32_t> tfLiteOutputShape  = tfLiteInterpreter.GetOutputShape(0);

    // Setup interpreter with Arm NN Delegate applied.
    auto armnnInterpreter = DelegateTestInterpreter(modelBuffer, backends);
    CHECK(armnnInterpreter.AllocateTensors() == kTfLiteOk);
    CHECK(armnnInterpreter.FillInputTensor<float>(inputValues, 0) == kTfLiteOk);
    CHECK(armnnInterpreter.Invoke() == kTfLiteOk);
    std::vector<float>   armnnOutputValues = armnnInterpreter.GetOutputResult<float>(0);
    std::vector<int32_t> armnnOutputShape  = armnnInterpreter.GetOutputShape(0);

    armnnDelegate::CompareOutputShape(tfLiteOutputShape, armnnOutputShape, outputShape);

    if (tensorType == ::tflite::TensorType_INT8)
    {
        // Allow 2% tolerance for Quantized weights
        armnnDelegate::CompareData(expectedOutputValues.data(), armnnOutputValues.data(),
                                   expectedOutputValues.size(), 2);
        armnnDelegate::CompareData(expectedOutputValues.data(), tfLiteOutputValues.data(),
                                   expectedOutputValues.size(), 2);
        armnnDelegate::CompareData(tfLiteOutputValues.data(), armnnOutputValues.data(),
                                   expectedOutputValues.size(), 2);
    }
    else
    {
        armnnDelegate::CompareOutputData<float>(tfLiteOutputValues, armnnOutputValues, expectedOutputValues);
    }

    tfLiteInterpreter.Cleanup();
    armnnInterpreter.Cleanup();
}

} // anonymous namespace