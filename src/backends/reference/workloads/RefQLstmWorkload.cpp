//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefQLstmWorkload.hpp"
#include "Activation.hpp"
#include "Encoders.hpp"
#include "Decoders.hpp"
#include "LstmUtils.hpp"
#include "RefWorkloadUtils.hpp"

namespace armnn
{

RefQLstmWorkload::RefQLstmWorkload(const QLstmQueueDescriptor &descriptor, const WorkloadInfo &info)
        : RefBaseWorkload<QLstmQueueDescriptor>(descriptor, info)
        , m_InputToInputWeightsTensor     (AssignScopedTensorHandle(descriptor.m_InputToInputWeights))
        , m_InputToForgetWeightsTensor    (AssignScopedTensorHandle(descriptor.m_InputToForgetWeights))
        , m_InputToCellWeightsTensor      (AssignScopedTensorHandle(descriptor.m_InputToCellWeights))
        , m_InputToOutputWeightsTensor    (AssignScopedTensorHandle(descriptor.m_InputToOutputWeights))

        , m_RecurrentToInputWeightsTensor (AssignScopedTensorHandle(descriptor.m_RecurrentToInputWeights))
        , m_RecurrentToForgetWeightsTensor(AssignScopedTensorHandle(descriptor.m_RecurrentToForgetWeights))
        , m_RecurrentToCellWeightsTensor  (AssignScopedTensorHandle(descriptor.m_RecurrentToCellWeights))
        , m_RecurrentToOutputWeightsTensor(AssignScopedTensorHandle(descriptor.m_RecurrentToOutputWeights))

        , m_CellToInputWeightsTensor      (AssignScopedTensorHandle(descriptor.m_CellToInputWeights))
        , m_CellToForgetWeightsTensor     (AssignScopedTensorHandle(descriptor.m_CellToForgetWeights))
        , m_CellToOutputWeightsTensor     (AssignScopedTensorHandle(descriptor.m_CellToOutputWeights))

        , m_InputGateBiasTensor           (AssignScopedTensorHandle(descriptor.m_InputGateBias))
        , m_ForgetGateBiasTensor          (AssignScopedTensorHandle(descriptor.m_ForgetGateBias))
        , m_CellBiasTensor                (AssignScopedTensorHandle(descriptor.m_CellBias))
        , m_OutputGateBiasTensor          (AssignScopedTensorHandle(descriptor.m_OutputGateBias))

        , m_ProjectionWeightsTensor       (AssignScopedTensorHandle(descriptor.m_ProjectionWeights))
        , m_ProjectionBiasTensor          (AssignScopedTensorHandle(descriptor.m_ProjectionBias))

        , m_InputLayerNormWeightsTensor   (AssignScopedTensorHandle(descriptor.m_InputLayerNormWeights))
        , m_ForgetLayerNormWeightsTensor  (AssignScopedTensorHandle(descriptor.m_ForgetLayerNormWeights))
        , m_CellLayerNormWeightsTensor    (AssignScopedTensorHandle(descriptor.m_CellLayerNormWeights))
        , m_OutputLayerNormWeightsTensor  (AssignScopedTensorHandle(descriptor.m_OutputLayerNormWeights))
{}

void RefQLstmWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefQLstmWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefQLstmWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    // This is a porting of the QLSTM::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs)
    // method in the Android code base
    // Note: this implementation wraps the arithmetic functions of the LSTM cell in Quantize/Dequantize ops, so all
    // computation is done in the floating point domain. Arithmetic functions are found in LstmUtils.cpp.
    // Refer to: android/frameworks/ml/nn/common/operations/QLSTM.cpp
    const DataType& internalType = armnn::DataType::QSymmS16;

    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
    const TensorInfo& outputStateInInfo = GetTensorInfo(inputs[1]);
    const TensorInfo& cellStateInInfo = GetTensorInfo(inputs[2]);

    const TensorInfo& outputStateOutInfo = GetTensorInfo(outputs[0]);
    const TensorInfo& cellStateOutInfo = GetTensorInfo(outputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[2]);

    const TensorShape& inputShape = inputInfo.GetShape();
    const TensorShape& outputStateInShape = outputStateInInfo.GetShape();
    const TensorShape& cellStateInShape = cellStateInInfo.GetShape();

    // Infer numBatches, inputSize, outputSize and numUnits
    const uint32_t numBatches = inputShape[0];
    const uint32_t inputSize  = inputShape[1];
    const uint32_t outputSize = outputStateInShape[1];
    const uint32_t numUnits   = cellStateInShape[1];

    // Optional param settings
    const bool cifgEnabled      = m_Data.m_Parameters.m_CifgEnabled;
    const bool peepholeEnabled  = m_Data.m_Parameters.m_PeepholeEnabled;
    const bool projectionEnabled = m_Data.m_Parameters.m_ProjectionEnabled;
    const bool layerNormEnabled = m_Data.m_Parameters.m_LayerNormEnabled;

    // Input decoders
    std::unique_ptr<Decoder<float>> inputDecoder =
            MakeDecoder<float>(inputInfo, inputs[0]->Map());
    std::unique_ptr<Decoder<float>> outputStateInDecoder =
            MakeDecoder<float>(outputStateInInfo, inputs[1]->Map());
    std::unique_ptr<Decoder<float>> cellStateInDecoder =
            MakeDecoder<float>(cellStateInInfo, inputs[2]->Map());

    // Output decoders
    std::unique_ptr<Decoder<float>> outputStateOutDecoder =
            MakeDecoder<float>(outputStateOutInfo, outputs[0]->Map());
    std::unique_ptr<Decoder<float>> cellStateOutDecoder =
            MakeDecoder<float>(cellStateOutInfo, outputs[1]->Map());
    std::unique_ptr<Decoder<float>> outputDecoder =
            MakeDecoder<float>(outputInfo, outputs[2]->Map());

    // Output encoders
    std::unique_ptr<Encoder<float>> outputStateOutEncoder =
            MakeEncoder<float>(outputStateOutInfo, outputs[0]->Map());
    std::unique_ptr<Encoder<float>> cellStateOutEncoder =
            MakeEncoder<float>(cellStateOutInfo, outputs[1]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder =
            MakeEncoder<float>(outputInfo, outputs[2]->Map());

    // Weights decoders
    std::unique_ptr<Decoder<float>> inputToForgetWeightsDecoder = MakeDecoder<float>(
            m_InputToForgetWeightsTensor->GetTensorInfo(), m_InputToForgetWeightsTensor->GetConstTensor<void>());
    std::unique_ptr<Decoder<float>> inputToCellWeightsDecoder = MakeDecoder<float>(
            m_InputToCellWeightsTensor->GetTensorInfo(), m_InputToCellWeightsTensor->GetConstTensor<void>());
    std::unique_ptr<Decoder<float>> inputToOutputWeightsDecoder = MakeDecoder<float>(
            m_InputToOutputWeightsTensor->GetTensorInfo(), m_InputToOutputWeightsTensor->GetConstTensor<void>());

    std::unique_ptr<Decoder<float>> recurrentToForgetWeightsDecoder = MakeDecoder<float>(
            m_RecurrentToForgetWeightsTensor->GetTensorInfo(),
            m_RecurrentToForgetWeightsTensor->GetConstTensor<void>());
    std::unique_ptr<Decoder<float>> recurrentToCellWeightsDecoder = MakeDecoder<float>(
            m_RecurrentToCellWeightsTensor->GetTensorInfo(), m_RecurrentToCellWeightsTensor->GetConstTensor<void>());
    std::unique_ptr<Decoder<float>> recurrentToOutputWeightsDecoder = MakeDecoder<float>(
            m_RecurrentToOutputWeightsTensor->GetTensorInfo(),
            m_RecurrentToOutputWeightsTensor->GetConstTensor<void>());

    // Optional CIFG params
    std::unique_ptr<Decoder<float>> inputToInputWeightsDecoder;
    std::unique_ptr<Decoder<float>> recurrentToInputWeightsDecoder;
    std::unique_ptr<Decoder<float>> inputGateBiasDecoder;

    // Optional Peephole params
    std::unique_ptr<Decoder<float>> cellToInputWeightsDecoder;
    std::unique_ptr<Decoder<float>> cellToForgetWeightsDecoder;
    std::unique_ptr<Decoder<float>> cellToOutputWeightsDecoder;

    // Optional Projection params
    std::unique_ptr<Decoder<float>> projectionWeightsDecoder;
    std::unique_ptr<Decoder<float>> projectionBiasDecoder;

    // Optional Layer Norm params
    std::unique_ptr<Decoder<float>> inputLayerNormWeightsDecoder;
    std::unique_ptr<Decoder<float>> forgetLayerNormWeightsDecoder;
    std::unique_ptr<Decoder<float>> cellLayerNormWeightsDecoder;
    std::unique_ptr<Decoder<float>> outputLayerNormWeightsDecoder;

    // Biases are only used when Layer Norm is enabled. Scale is defined as (XLayerNormWeights Scale / 1024)
    std::unique_ptr<Decoder<float>> forgetGateBiasDecoder;
    std::unique_ptr<Decoder<float>> cellGateBiasDecoder;
    std::unique_ptr<Decoder<float>> outputGateBiasDecoder;

    // Int16 vectors for internal state data (to be decoded/encoded)
    const uint32_t stateTensorSize = numBatches * numUnits;
    std::vector<int16_t> inputGateData(stateTensorSize);
    std::vector<int16_t> cellGateData(stateTensorSize);
    std::vector<int16_t> forgetGateData(stateTensorSize);
    std::vector<int16_t> outputGateData(stateTensorSize);
    std::vector<int32_t> hiddenStateData(stateTensorSize);
    std::vector<int16_t> outputInt16Data(numBatches * outputSize);

    armnn::TensorInfo inputGateInfo(
            {numBatches , numUnits}, armnn::DataType::QSymmS16, m_Data.m_Parameters.m_InputIntermediateScale, 0);
    armnn::TensorInfo cellGateInfo(
            {numBatches , numUnits}, armnn::DataType::QSymmS16, m_Data.m_Parameters.m_CellIntermediateScale, 0);
    armnn::TensorInfo forgetGateInfo(
            {numBatches , numUnits}, armnn::DataType::QSymmS16, m_Data.m_Parameters.m_ForgetIntermediateScale, 0);
    armnn::TensorInfo outputGateInfo(
            {numBatches , numUnits}, armnn::DataType::QSymmS16, m_Data.m_Parameters.m_OutputIntermediateScale, 0);
    armnn::TensorInfo hiddenStateInfo({numBatches, numUnits},
                                      armnn::DataType::QAsymmS8,
                                      m_Data.m_Parameters.m_HiddenStateScale,
                                      m_Data.m_Parameters.m_HiddenStateZeroPoint);
    armnn::TensorInfo outputInt16Info({numBatches , outputSize},
                                      armnn::DataType::QSymmS16,
                                      outputInfo.GetQuantizationScale(),
                                      outputInfo.GetQuantizationOffset());

    // Decoders/Encoders for internal states
    std::unique_ptr<Decoder<float>> inputGateDecoder =
            MakeDecoder<float>(inputGateInfo, inputGateData.data());
    std::unique_ptr<Decoder<float>> cellGateDecoder =
            MakeDecoder<float>(cellGateInfo, cellGateData.data());
    std::unique_ptr<Decoder<float>> forgetGateDecoder =
            MakeDecoder<float>(forgetGateInfo, forgetGateData.data());
    std::unique_ptr<Decoder<float>> outputGateDecoder =
            MakeDecoder<float>(outputGateInfo, outputGateData.data());
    std::unique_ptr<Decoder<float>> hiddenStateDecoder =
            MakeDecoder<float>(hiddenStateInfo, hiddenStateData.data());

    std::unique_ptr<Encoder<float>> inputGateEncoder =
            MakeEncoder<float>(inputGateInfo, inputGateData.data());
    std::unique_ptr<Encoder<float>> cellGateEncoder =
            MakeEncoder<float>(cellGateInfo, cellGateData.data());
    std::unique_ptr<Encoder<float>> forgetGateEncoder =
            MakeEncoder<float>(forgetGateInfo, forgetGateData.data());
    std::unique_ptr<Encoder<float>> outputGateEncoder =
            MakeEncoder<float>(outputGateInfo, outputGateData.data());
    std::unique_ptr<Encoder<float>> hiddenStateEncoder =
            MakeEncoder<float>(hiddenStateInfo, hiddenStateData.data());

    // Int16 used to accumulate output to prevent overflowing (after Projection MatMul)
    std::unique_ptr<Decoder<float>> outputInt16Decoder =
            MakeDecoder<float>(outputInt16Info, outputInt16Data.data());
    std::unique_ptr<Encoder<float>> outputInt16Encoder =
            MakeEncoder<float>(outputInt16Info, outputInt16Data.data());

    // Create decoders for optional params if they are enabled
    if (!cifgEnabled)
    {
        inputToInputWeightsDecoder = MakeDecoder<float>(
                m_InputToInputWeightsTensor->GetTensorInfo(), m_InputToInputWeightsTensor->GetConstTensor<void>());
        recurrentToInputWeightsDecoder = MakeDecoder<float>(m_RecurrentToInputWeightsTensor->GetTensorInfo(),
                                                            m_RecurrentToInputWeightsTensor->GetConstTensor<void>());
    }

    if (peepholeEnabled)
    {
        if (!cifgEnabled)
        {
            cellToInputWeightsDecoder = MakeDecoder<float>(
                    m_CellToInputWeightsTensor->GetTensorInfo(), m_CellToInputWeightsTensor->GetConstTensor<void>());
        }
        cellToForgetWeightsDecoder = MakeDecoder<float>(
                m_CellToForgetWeightsTensor->GetTensorInfo(), m_CellToForgetWeightsTensor->GetConstTensor<void>());
        cellToOutputWeightsDecoder = MakeDecoder<float>(
                m_CellToOutputWeightsTensor->GetTensorInfo(), m_CellToOutputWeightsTensor->GetConstTensor<void>());
    }

    if (projectionEnabled)
    {
        projectionWeightsDecoder = MakeDecoder<float>(
                m_ProjectionWeightsTensor->GetTensorInfo(), m_ProjectionWeightsTensor->GetConstTensor<void>());
        if (m_ProjectionBiasTensor)
        {
            projectionBiasDecoder = MakeDecoder<float>(
                    m_ProjectionBiasTensor->GetTensorInfo(), m_ProjectionBiasTensor->GetConstTensor<void>());
        }
    }

    if (layerNormEnabled)
    {
        if (!cifgEnabled)
        {
            inputLayerNormWeightsDecoder = MakeDecoder<float>(m_InputLayerNormWeightsTensor->GetTensorInfo(),
                                                              m_InputLayerNormWeightsTensor->GetConstTensor<void>());

            // Bias only used if layer norm enabled
            armnn::TensorInfo inputGateBiasTensorInfo({outputSize}, armnn::DataType::Signed32,
                    m_InputLayerNormWeightsTensor->GetTensorInfo().GetQuantizationScale() / 1024, 0);
            inputGateBiasDecoder = MakeDecoder<float>(
                    inputGateBiasTensorInfo, m_InputGateBiasTensor->GetConstTensor<void>());
        }

        forgetLayerNormWeightsDecoder = MakeDecoder<float>(
                m_ForgetLayerNormWeightsTensor->GetTensorInfo(),
                m_ForgetLayerNormWeightsTensor->GetConstTensor<void>());
        cellLayerNormWeightsDecoder = MakeDecoder<float>(
                m_CellLayerNormWeightsTensor->GetTensorInfo(), m_CellLayerNormWeightsTensor->GetConstTensor<void>());
        outputLayerNormWeightsDecoder = MakeDecoder<float>(
                m_OutputLayerNormWeightsTensor->GetTensorInfo(),
                m_OutputLayerNormWeightsTensor->GetConstTensor<void>());

        // Bias only used if layer norm enabled
        armnn::TensorInfo forgetGateBiasTensorInfo({outputSize}, armnn::DataType::Signed32,
                m_ForgetLayerNormWeightsTensor->GetTensorInfo().GetQuantizationScale() / 1024, 0);
        forgetGateBiasDecoder = MakeDecoder<float>(
                forgetGateBiasTensorInfo, m_ForgetGateBiasTensor->GetConstTensor<void>());

        armnn::TensorInfo cellGateBiasTensorInfo({outputSize}, armnn::DataType::Signed32,
                m_CellLayerNormWeightsTensor->GetTensorInfo().GetQuantizationScale() / 1024, 0);
        cellGateBiasDecoder = MakeDecoder<float>(
                cellGateBiasTensorInfo, m_CellBiasTensor->GetConstTensor<void>());

        armnn::TensorInfo outputGateBiasTensorInfo({outputSize}, armnn::DataType::Signed32,
                m_OutputLayerNormWeightsTensor->GetTensorInfo().GetQuantizationScale() / 1024, 0);
        outputGateBiasDecoder = MakeDecoder<float>(
                outputGateBiasTensorInfo, m_OutputGateBiasTensor->GetConstTensor<void>());
    }

    // Initialize internal state tensors with zeroes.
    if (!cifgEnabled)
    {
        ZeroVector(*inputGateEncoder, stateTensorSize);
    }
    ZeroVector(*forgetGateEncoder, stateTensorSize);
    ZeroVector(*cellGateEncoder, stateTensorSize);
    ZeroVector(*outputGateEncoder, stateTensorSize);
    ZeroVector(*hiddenStateEncoder, stateTensorSize);

    // Input weights * Input
    if (!cifgEnabled)
    {
        MatrixBatchVectorMultiplyAccumulate(*inputToInputWeightsDecoder,
                                            numUnits, inputSize, *inputDecoder, numBatches, *inputGateEncoder);
    }

    MatrixBatchVectorMultiplyAccumulate(*inputToForgetWeightsDecoder,
                                        numUnits, inputSize, *inputDecoder, numBatches, *forgetGateEncoder);

    MatrixBatchVectorMultiplyAccumulate(*inputToCellWeightsDecoder,
                                        numUnits, inputSize, *inputDecoder, numBatches, *cellGateEncoder);

    MatrixBatchVectorMultiplyAccumulate(*inputToOutputWeightsDecoder,
                                        numUnits, inputSize, *inputDecoder, numBatches, *outputGateEncoder);

    // Recurrent weights * OutputStateIn
    if (!cifgEnabled)
    {
        MatrixBatchVectorMultiplyAccumulate(*recurrentToInputWeightsDecoder,
                                            numUnits, outputSize, *outputStateInDecoder, numBatches, *inputGateEncoder);
    }

    MatrixBatchVectorMultiplyAccumulate(*recurrentToForgetWeightsDecoder,
                                        numUnits, outputSize, *outputStateInDecoder, numBatches, *forgetGateEncoder);

    MatrixBatchVectorMultiplyAccumulate(*recurrentToCellWeightsDecoder,
                                        numUnits, outputSize, *outputStateInDecoder, numBatches, *cellGateEncoder);

    MatrixBatchVectorMultiplyAccumulate(*recurrentToOutputWeightsDecoder,
                                        numUnits, outputSize, *outputStateInDecoder, numBatches, *outputGateEncoder);

    // Input gate.
    if (!cifgEnabled)
    {
        if (peepholeEnabled)
        {
            VectorBatchVectorCwiseProductAccumulate(*cellToInputWeightsDecoder,
                                                    numUnits, *cellStateInDecoder, numBatches, *inputGateEncoder);
        }

        if (layerNormEnabled)
        {
            inputGateInfo.SetQuantizationScale(inputInfo.GetQuantizationScale() *
                                               m_InputLayerNormWeightsTensor->GetTensorInfo().GetQuantizationScale() *
                                               1024);
            inputGateEncoder = MakeEncoder<float>(inputGateInfo, inputGateData.data());

            MeanStddevNormalization(*inputGateDecoder,
                                    *inputGateEncoder, numUnits, numBatches, m_LayerNormEpsilon);

            inputGateDecoder = MakeDecoder<float>(inputGateInfo, inputGateData.data());

            VectorBatchVectorCwiseProduct(*inputLayerNormWeightsDecoder,
                                          numUnits, *inputGateDecoder, numBatches, *inputGateEncoder);

            inputGateInfo.SetQuantizationScale(1.f / 4096);
            inputGateEncoder = MakeEncoder<float>(inputGateInfo, inputGateData.data());

            VectorBatchVectorAdd(*inputGateBiasDecoder,
                                 numUnits, *inputGateDecoder, numBatches, *inputGateEncoder);

            inputGateDecoder = MakeDecoder<float>(inputGateInfo, inputGateData.data());
        }

        inputGateInfo.SetQuantizationScale(cellStateOutInfo.GetQuantizationScale());
        inputGateEncoder = MakeEncoder<float>(inputGateInfo, inputGateData.data());

        // Input gate sigmoid
        Activation(*inputGateDecoder, *inputGateEncoder,
                   TensorInfo({numUnits, numBatches}, internalType),
                   ActivationFunction::Sigmoid, 0, 0);

        inputGateDecoder = MakeDecoder<float>(inputGateInfo, inputGateData.data());
    }

    // Forget gate
    if (peepholeEnabled)
    {
        VectorBatchVectorCwiseProductAccumulate(*cellToForgetWeightsDecoder, numUnits,
                                                *cellStateInDecoder, numBatches, *forgetGateEncoder);
    }

    if (layerNormEnabled)
    {
        // Quantize layer norm output to Input Scale * m_ForgetLayerNormWeightsTensor * 1024
        forgetGateInfo.SetQuantizationScale(inputInfo.GetQuantizationScale() *
                                            m_ForgetLayerNormWeightsTensor->GetTensorInfo().GetQuantizationScale() *
                                            1024);
        forgetGateEncoder = MakeEncoder<float>(forgetGateInfo, forgetGateData.data());



        MeanStddevNormalization(*forgetGateDecoder,
                                *forgetGateEncoder, numUnits, numBatches, m_LayerNormEpsilon);


        forgetGateDecoder = MakeDecoder<float>(forgetGateInfo, forgetGateData.data());

        VectorBatchVectorCwiseProduct(*forgetLayerNormWeightsDecoder,
                                      numUnits, *forgetGateDecoder, numBatches, *forgetGateEncoder);


        // Dequantize layer norm output to (1 / 4096)
        forgetGateInfo.SetQuantizationScale(1.f / 4096);
        forgetGateEncoder = MakeEncoder<float>(forgetGateInfo, forgetGateData.data());

        VectorBatchVectorAdd(*forgetGateBiasDecoder,
                             numUnits, *forgetGateDecoder, numBatches, *forgetGateEncoder);


        forgetGateDecoder = MakeDecoder<float>(forgetGateInfo, forgetGateData.data());
    }

    forgetGateInfo.SetQuantizationScale(cellStateOutInfo.GetQuantizationScale());
    forgetGateEncoder = MakeEncoder<float>(forgetGateInfo, forgetGateData.data());

    // Forget gate sigmoid
    Activation(*forgetGateDecoder, *forgetGateEncoder,
               TensorInfo({numUnits, numBatches}, internalType),
               ActivationFunction::Sigmoid, 0, 0);

    forgetGateDecoder = MakeDecoder<float>(forgetGateInfo, forgetGateData.data());

    // Cell (Modulation) gate
    if (layerNormEnabled)
    {
        cellGateInfo.SetQuantizationScale(inputInfo.GetQuantizationScale() *
                                          m_CellLayerNormWeightsTensor->GetTensorInfo().GetQuantizationScale() *
                                          1024);
        cellGateEncoder = MakeEncoder<float>(cellGateInfo, cellGateData.data());

        MeanStddevNormalization(*cellGateDecoder, *cellGateEncoder, numUnits, numBatches, m_LayerNormEpsilon);

        cellGateDecoder = MakeDecoder<float>(cellGateInfo, cellGateData.data());

        VectorBatchVectorCwiseProduct(*cellLayerNormWeightsDecoder,
                                      numUnits, *cellGateDecoder, numBatches, *cellGateEncoder);

        cellGateInfo.SetQuantizationScale(1.f / 4096);
        cellGateEncoder = MakeEncoder<float>(cellGateInfo, cellGateData.data());

        VectorBatchVectorAdd(*cellGateBiasDecoder,
                             numUnits, *cellGateDecoder, numBatches, *cellGateEncoder);

        cellGateDecoder = MakeDecoder<float>(cellGateInfo, cellGateData.data());
    }

    cellGateInfo.SetQuantizationScale(cellStateOutInfo.GetQuantizationScale());
    cellGateEncoder = MakeEncoder<float>(cellGateInfo, cellGateData.data());

    // Cell (Modulation) gate tanH
    Activation(*cellGateDecoder, *cellGateEncoder,
               TensorInfo({numUnits, numBatches}, internalType),
               ActivationFunction::TanH, 1.0f, 1.0f);

    cellGateDecoder = MakeDecoder<float>(cellGateInfo, cellGateData.data());

    VectorVectorCwiseProduct(*forgetGateDecoder, *cellStateInDecoder, stateTensorSize, *cellStateOutEncoder);

    if (cifgEnabled)
    {
        Sub1Vector(*forgetGateDecoder, stateTensorSize, *forgetGateEncoder);
        VectorVectorCwiseProductAccumulate(
                *cellGateDecoder, *forgetGateDecoder, stateTensorSize, *cellStateOutEncoder);
    }
    else
    {
        VectorVectorCwiseProductAccumulate(
                *cellGateDecoder, *inputGateDecoder, stateTensorSize, *cellStateOutEncoder);
    }

    // Final cell state out calculated here
    if (m_Data.m_Parameters.m_CellClip > 0.0)
    {
        ClipVector(*cellStateOutDecoder, stateTensorSize, m_Data.m_Parameters.m_CellClip, *cellStateOutEncoder);
    }

    // Output gate.
    if (peepholeEnabled)
    {
        VectorBatchVectorCwiseProductAccumulate(*cellToOutputWeightsDecoder,
                                                numUnits, *cellStateOutDecoder, numBatches, *outputGateEncoder);
    }

    if (layerNormEnabled)
    {
        outputGateInfo.SetQuantizationScale(inputInfo.GetQuantizationScale() *
                                            m_OutputLayerNormWeightsTensor->GetTensorInfo().GetQuantizationScale() *
                                            1024);
        outputGateEncoder = MakeEncoder<float>(outputGateInfo, outputGateData.data());

        MeanStddevNormalization(*outputGateDecoder, *outputGateEncoder, numUnits, numBatches, m_LayerNormEpsilon);

        outputGateDecoder = MakeDecoder<float>(outputGateInfo, outputGateData.data());

        VectorBatchVectorCwiseProduct(*outputLayerNormWeightsDecoder, numUnits, *outputGateDecoder,
                                      numBatches, *outputGateEncoder);

        outputGateInfo.SetQuantizationScale(1.f / 4096);
        outputGateEncoder = MakeEncoder<float>(outputGateInfo, outputGateData.data());

        VectorBatchVectorAdd(*outputGateBiasDecoder, numUnits, *outputGateDecoder, numBatches, *outputGateEncoder);

        outputGateDecoder = MakeDecoder<float>(outputGateInfo, outputGateData.data());
    }

    outputGateInfo.SetQuantizationScale(cellStateOutInfo.GetQuantizationScale());
    outputGateEncoder = MakeEncoder<float>(outputGateInfo, outputGateData.data());

    // Output gate sigmoid
    Activation(*outputGateDecoder, *outputGateEncoder,
               TensorInfo({numUnits, numBatches}, internalType),
               ActivationFunction::Sigmoid, 0, 0);

    outputGateDecoder = MakeDecoder<float>(outputGateInfo, outputGateData.data());

    // Hidden state tanH
    Activation(*cellStateOutDecoder, *cellGateEncoder,
               TensorInfo({numUnits, numBatches}, internalType),
               ActivationFunction::TanH, 1.0f, 1.0f);

    // Final hidden state output
    VectorVectorCwiseProduct(*outputGateDecoder, *cellGateDecoder, stateTensorSize, *hiddenStateEncoder);

    // Projection
    if (m_Data.m_Parameters.m_ProjectionEnabled)
    {
        if (m_ProjectionBiasTensor)
        {
            VectorBatchVectorAssign(*projectionBiasDecoder, outputSize, numBatches, *outputInt16Encoder);
        }

        MatrixBatchVectorMultiplyAccumulate(*projectionWeightsDecoder, outputSize, numUnits, *hiddenStateDecoder,
                                            numBatches, *outputInt16Encoder);

        CopyVector(*outputInt16Decoder, numBatches * outputSize, *outputEncoder);

        if (m_Data.m_Parameters.m_ProjectionClip > 0.0)
        {
            ClipVector(*outputDecoder, numBatches * outputSize, m_Data.m_Parameters.m_ProjectionClip, *outputEncoder);
        }
    }
    else
    {
        // Output has same quantization scale as hidden state if projection is disabled
        CopyVector(*hiddenStateDecoder, numBatches * outputSize, *outputEncoder);
    }

    // output == outputStateOut
    CopyVector(*outputDecoder, numBatches * outputSize, *outputStateOutEncoder);
}

} //namespace armnn
