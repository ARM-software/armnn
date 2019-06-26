//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefLstmWorkload.hpp"
#include "Activation.hpp"
#include "Encoders.hpp"
#include "Decoders.hpp"
#include "LstmUtils.hpp"
#include "RefWorkloadUtils.hpp"

namespace armnn
{

RefLstmWorkload::RefLstmWorkload(const LstmQueueDescriptor &descriptor, const WorkloadInfo &info)
    : BaseWorkload<LstmQueueDescriptor>(descriptor, info)
    , m_InputToInputWeightsTensor     (AssignScopedCpuTensorHandle(descriptor.m_InputToInputWeights))
    , m_InputToForgetWeightsTensor    (AssignScopedCpuTensorHandle(descriptor.m_InputToForgetWeights))
    , m_InputToCellWeightsTensor      (AssignScopedCpuTensorHandle(descriptor.m_InputToCellWeights))
    , m_InputToOutputWeightsTensor    (AssignScopedCpuTensorHandle(descriptor.m_InputToOutputWeights))
    , m_RecurrentToInputWeightsTensor (AssignScopedCpuTensorHandle(descriptor.m_RecurrentToInputWeights))
    , m_RecurrentToForgetWeightsTensor(AssignScopedCpuTensorHandle(descriptor.m_RecurrentToForgetWeights))
    , m_RecurrentToCellWeightsTensor  (AssignScopedCpuTensorHandle(descriptor.m_RecurrentToCellWeights))
    , m_RecurrentToOutputWeightsTensor(AssignScopedCpuTensorHandle(descriptor.m_RecurrentToOutputWeights))
    , m_CellToInputWeightsTensor      (AssignScopedCpuTensorHandle(descriptor.m_CellToInputWeights))
    , m_CellToForgetWeightsTensor     (AssignScopedCpuTensorHandle(descriptor.m_CellToForgetWeights))
    , m_CellToOutputWeightsTensor     (AssignScopedCpuTensorHandle(descriptor.m_CellToOutputWeights))
    , m_InputGateBiasTensor           (AssignScopedCpuTensorHandle(descriptor.m_InputGateBias))
    , m_ForgetGateBiasTensor          (AssignScopedCpuTensorHandle(descriptor.m_ForgetGateBias))
    , m_CellBiasTensor                (AssignScopedCpuTensorHandle(descriptor.m_CellBias))
    , m_OutputGateBiasTensor          (AssignScopedCpuTensorHandle(descriptor.m_OutputGateBias))
    , m_ProjectionWeightsTensor       (AssignScopedCpuTensorHandle(descriptor.m_ProjectionWeights))
    , m_ProjectionBiasTensor          (AssignScopedCpuTensorHandle(descriptor.m_ProjectionBias))
    , m_InputLayerNormWeights         (AssignScopedCpuTensorHandle(descriptor.m_InputLayerNormWeights))
    , m_ForgetLayerNormWeights        (AssignScopedCpuTensorHandle(descriptor.m_ForgetLayerNormWeights))
    , m_CellLayerNormWeights          (AssignScopedCpuTensorHandle(descriptor.m_CellLayerNormWeights))
    , m_OutputLayerNormWeights        (AssignScopedCpuTensorHandle(descriptor.m_OutputLayerNormWeights))
{}

void RefLstmWorkload::Execute() const
{
    // This is a porting of the LSTM::Eval() method in the Android code base
    // Refer to: android/frameworks/ml/nn/common/operations/LSTM.cpp

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    const TensorShape& inputShape = inputInfo.GetShape();
    const DataType& outputType = outputInfo.GetDataType();

    std::unique_ptr<Encoder<float>> outputStateOut = MakeEncoder<float>(outputInfo, m_Data.m_Outputs[1]->Map());
    std::unique_ptr<Encoder<float>> cellStateOut   = MakeEncoder<float>(outputInfo, m_Data.m_Outputs[2]->Map());
    std::unique_ptr<Encoder<float>> output         = MakeEncoder<float>(outputInfo, m_Data.m_Outputs[3]->Map());

    std::unique_ptr<Decoder<float>> cellStateOutDecoder = MakeDecoder<float>(outputInfo, m_Data.m_Outputs[2]->Map());
    std::unique_ptr<Decoder<float>> outputDecoder       = MakeDecoder<float>(outputInfo, m_Data.m_Outputs[3]->Map());

    std::unique_ptr<Decoder<float>> inputData     = MakeDecoder<float>(inputInfo, m_Data.m_Inputs[0]->Map());
    std::unique_ptr<Decoder<float>> outputStateIn = MakeDecoder<float>(inputInfo, m_Data.m_Inputs[1]->Map());
    std::unique_ptr<Decoder<float>> cellStateIn   = MakeDecoder<float>(inputInfo, m_Data.m_Inputs[2]->Map());

    const uint32_t nBatch = inputShape[0];
    const uint32_t nInput = inputShape[1];

    const uint32_t nCell   = m_InputToOutputWeightsTensor->GetShape()[0];
    const uint32_t nOutput = m_RecurrentToOutputWeightsTensor->GetShape()[1];

    const bool useCifg      = m_Data.m_Parameters.m_CifgEnabled;
    const bool usePeephole  = m_Data.m_Parameters.m_PeepholeEnabled;
    const bool useLayerNorm = m_Data.m_Parameters.m_LayerNormEnabled;

    // Index the scratch buffers pointers to the global scratch buffer.
    std::unique_ptr<Encoder<float>> inputGateScratch  = MakeEncoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());
    std::unique_ptr<Encoder<float>> cellScratch       = MakeEncoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());
    std::unique_ptr<Encoder<float>> forgetGateScratch = MakeEncoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());
    std::unique_ptr<Encoder<float>> outputGateScratch = MakeEncoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());

    std::unique_ptr<Decoder<float>> inputGateScratchDecoder =
        MakeDecoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());
    std::unique_ptr<Decoder<float>> cellScratchDecoder =
        MakeDecoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());
    std::unique_ptr<Decoder<float>> forgetGateScratchDecoder =
        MakeDecoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());
    std::unique_ptr<Decoder<float>> outputGateScratchDecoder =
        MakeDecoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());

    if (useCifg)
    {
        *cellScratch       += (0 * nCell * nBatch);
        *forgetGateScratch += (1 * nCell * nBatch);
        *outputGateScratch += (2 * nCell * nBatch);

        *cellScratchDecoder       += (0 * nCell * nBatch);
        *forgetGateScratchDecoder += (1 * nCell * nBatch);
        *outputGateScratchDecoder += (2 * nCell * nBatch);
    }
    else
    {
        *inputGateScratch  += (0 * nCell * nBatch);
        *cellScratch       += (1 * nCell * nBatch);
        *forgetGateScratch += (2 * nCell * nBatch);
        *outputGateScratch += (3 * nCell * nBatch);

        *inputGateScratchDecoder  += (0 * nCell * nBatch);
        *cellScratchDecoder       += (1 * nCell * nBatch);
        *forgetGateScratchDecoder += (2 * nCell * nBatch);
        *outputGateScratchDecoder += (3 * nCell * nBatch);
    }

    std::unique_ptr<Decoder<float>> inputToInputWeightsTensor;
    std::unique_ptr<Decoder<float>> inputToForgetWeightsTensor = MakeDecoder<float>(
        m_InputToForgetWeightsTensor->GetTensorInfo(), m_InputToForgetWeightsTensor->GetTensor<void>());
    std::unique_ptr<Decoder<float>> inputToCellWeightsTensor = MakeDecoder<float>(
        m_InputToCellWeightsTensor->GetTensorInfo(), m_InputToCellWeightsTensor->GetTensor<void>());
    std::unique_ptr<Decoder<float>> inputToOutputWeightsTensor = MakeDecoder<float>(
        m_InputToOutputWeightsTensor->GetTensorInfo(), m_InputToOutputWeightsTensor->GetTensor<void>());

    std::unique_ptr<Decoder<float>> recurrentToInputWeightsTensor;
    std::unique_ptr<Decoder<float>> recurrentToForgetWeightsTensor = MakeDecoder<float>(
        m_RecurrentToForgetWeightsTensor->GetTensorInfo(), m_RecurrentToForgetWeightsTensor->GetTensor<void>());
    std::unique_ptr<Decoder<float>> recurrentToCellWeightsTensor = MakeDecoder<float>(
        m_RecurrentToCellWeightsTensor->GetTensorInfo(), m_RecurrentToCellWeightsTensor->GetTensor<void>());
    std::unique_ptr<Decoder<float>> recurrentToOutputWeightsTensor = MakeDecoder<float>(
        m_RecurrentToOutputWeightsTensor->GetTensorInfo(), m_RecurrentToOutputWeightsTensor->GetTensor<void>());

    std::unique_ptr<Decoder<float>> inputGateBiasTensor;
    std::unique_ptr<Decoder<float>> forgetGateBiasTensor = MakeDecoder<float>(
        m_ForgetGateBiasTensor->GetTensorInfo(), m_ForgetGateBiasTensor->GetTensor<void>());
    std::unique_ptr<Decoder<float>> cellBiasTensor = MakeDecoder<float>(
        m_CellBiasTensor->GetTensorInfo(), m_CellBiasTensor->GetTensor<void>());
    std::unique_ptr<Decoder<float>> outputGateBiasTensor = MakeDecoder<float>(
        m_OutputGateBiasTensor->GetTensorInfo(), m_OutputGateBiasTensor->GetTensor<void>());

    std::unique_ptr<Decoder<float>> cellToInputWeightsTensor;
    std::unique_ptr<Decoder<float>> cellToForgetWeightsTensor;
    std::unique_ptr<Decoder<float>> cellToOutputWeightsTensor;

    std::unique_ptr<Decoder<float>> projectionWeightsTensor;
    std::unique_ptr<Decoder<float>> projectionBiasTensor;

    std::unique_ptr<Decoder<float>> inputLayerNormWeights;
    std::unique_ptr<Decoder<float>> forgetLayerNormWeights;
    std::unique_ptr<Decoder<float>> cellLayerNormWeights;
    std::unique_ptr<Decoder<float>> outputLayerNormWeights;

    if (useLayerNorm)
    {
        if (!useCifg)
        {
            inputLayerNormWeights = MakeDecoder<float>(
                    m_InputLayerNormWeights->GetTensorInfo(), m_InputLayerNormWeights->GetTensor<void>());
        }
        forgetLayerNormWeights = MakeDecoder<float>(
                m_ForgetLayerNormWeights->GetTensorInfo(), m_ForgetLayerNormWeights->GetTensor<void>());
        cellLayerNormWeights = MakeDecoder<float>(
                m_CellLayerNormWeights->GetTensorInfo(), m_CellLayerNormWeights->GetTensor<void>());
        outputLayerNormWeights = MakeDecoder<float>(
                m_OutputLayerNormWeights->GetTensorInfo(), m_OutputLayerNormWeights->GetTensor<void>());
    }

    if (!useCifg)
    {
        inputToInputWeightsTensor = MakeDecoder<float>(
            m_InputToInputWeightsTensor->GetTensorInfo(), m_InputToInputWeightsTensor->GetTensor<void>());
        inputGateBiasTensor = MakeDecoder<float>(
            m_InputGateBiasTensor->GetTensorInfo(), m_InputGateBiasTensor->GetTensor<void>());
        recurrentToInputWeightsTensor = MakeDecoder<float>(
            m_RecurrentToInputWeightsTensor->GetTensorInfo(), m_RecurrentToInputWeightsTensor->GetTensor<void>());
    }

    if (usePeephole)
    {
        cellToForgetWeightsTensor = MakeDecoder<float>(
            m_CellToForgetWeightsTensor->GetTensorInfo(), m_CellToForgetWeightsTensor->GetTensor<void>());
        cellToOutputWeightsTensor = MakeDecoder<float>(
            m_CellToOutputWeightsTensor->GetTensorInfo(), m_CellToOutputWeightsTensor->GetTensor<void>());
    }

    if (!useCifg && usePeephole)
    {
        cellToInputWeightsTensor = MakeDecoder<float>(
            m_CellToInputWeightsTensor->GetTensorInfo(), m_CellToInputWeightsTensor->GetTensor<void>());
    }

    if (m_Data.m_Parameters.m_ProjectionEnabled)
    {
        projectionWeightsTensor = MakeDecoder<float>(
            m_ProjectionWeightsTensor->GetTensorInfo(), m_ProjectionWeightsTensor->GetTensor<void>());
        if (m_ProjectionBiasTensor)
        {
            projectionBiasTensor = MakeDecoder<float>(
                m_ProjectionBiasTensor->GetTensorInfo(), m_ProjectionBiasTensor->GetTensor<void>());
        }
    }

    if (!useLayerNorm)
    {
        // Initialize scratch buffers with bias.
        if (!useCifg)
        {
            VectorBatchVectorAssign(*inputGateBiasTensor,
                                    nCell, nBatch, *inputGateScratch);
        }
        VectorBatchVectorAssign(*forgetGateBiasTensor,
                                nCell, nBatch, *forgetGateScratch);
        VectorBatchVectorAssign(*cellBiasTensor,
                                nCell, nBatch, *cellScratch);
        VectorBatchVectorAssign(*outputGateBiasTensor,
                                nCell, nBatch, *outputGateScratch);
    }
    else
    {
        // Initialize scratch buffers with zeroes.
        if (!useCifg)
        {
            ZeroVector(*inputGateScratch, nCell * nBatch);
        }
        ZeroVector(*forgetGateScratch, nCell * nBatch);
        ZeroVector(*cellScratch      , nCell * nBatch);
        ZeroVector(*outputGateScratch, nCell * nBatch);
    }

    // For each batch and cell: compute input_weight * input.
    if (!useCifg)
    {
        MatrixBatchVectorMultiplyAccumulate(*inputToInputWeightsTensor,
                                            nCell, nInput, *inputData, nBatch, *inputGateScratch);
    }
    MatrixBatchVectorMultiplyAccumulate(*inputToForgetWeightsTensor,
                                        nCell, nInput, *inputData, nBatch, *forgetGateScratch);
    MatrixBatchVectorMultiplyAccumulate(*inputToCellWeightsTensor,
                                        nCell, nInput, *inputData, nBatch, *cellScratch);
    MatrixBatchVectorMultiplyAccumulate(*inputToOutputWeightsTensor,
                                        nCell, nInput, *inputData, nBatch, *outputGateScratch);

    // For each batch and cell: compute recurrent_weight * output_state.
    if (!useCifg)
    {
        MatrixBatchVectorMultiplyAccumulate(*recurrentToInputWeightsTensor,
                                            nCell, nOutput, *outputStateIn, nBatch, *inputGateScratch);
    }
    MatrixBatchVectorMultiplyAccumulate(*recurrentToForgetWeightsTensor,
                                        nCell, nOutput, *outputStateIn, nBatch, *forgetGateScratch);
    MatrixBatchVectorMultiplyAccumulate(*recurrentToCellWeightsTensor,
                                        nCell, nOutput, *outputStateIn, nBatch, *cellScratch);
    MatrixBatchVectorMultiplyAccumulate(*recurrentToOutputWeightsTensor,
                                        nCell, nOutput, *outputStateIn, nBatch, *outputGateScratch);

    // For each batch and cell: update input gate.
    if (!useCifg)
    {
        if (usePeephole)
        {
            VectorBatchVectorCwiseProductAccumulate(*cellToInputWeightsTensor,
                                                    nCell, *cellStateIn, nBatch, *inputGateScratch);
        }
        if (useLayerNorm)
        {
            MeanStddevNormalization(*inputGateScratchDecoder,
                                    *inputGateScratch, nCell, nBatch, m_LayerNormEpsilon);
            VectorBatchVectorCwiseProduct(*inputLayerNormWeights,
                                          nCell, *inputGateScratchDecoder, nBatch, *inputGateScratch);
            VectorBatchVectorAdd(*inputGateBiasTensor,
                                 nCell, *inputGateScratchDecoder, nBatch, *inputGateScratch);
        }
        Activation(*inputGateScratchDecoder, *inputGateScratch,
                   TensorInfo({nCell, nBatch}, outputType),
                   ActivationFunction::Sigmoid, 0, 0);
    }

    // For each batch and cell: update forget gate.
    if (usePeephole)
    {
        VectorBatchVectorCwiseProductAccumulate(*cellToForgetWeightsTensor, nCell,
                                                *cellStateIn, nBatch, *forgetGateScratch);
    }
    if (useLayerNorm)
    {
        MeanStddevNormalization(*forgetGateScratchDecoder,
                                *forgetGateScratch, nCell, nBatch, m_LayerNormEpsilon);
        VectorBatchVectorCwiseProduct(*forgetLayerNormWeights,
                                      nCell, *forgetGateScratchDecoder, nBatch, *forgetGateScratch);
        VectorBatchVectorAdd(*forgetGateBiasTensor,
                             nCell, *forgetGateScratchDecoder, nBatch, *forgetGateScratch);
    }
    Activation(*forgetGateScratchDecoder, *forgetGateScratch,
               TensorInfo({nCell, nBatch}, outputType),
               ActivationFunction::Sigmoid, 0, 0);

    // For each batch and cell: update the cell.
    if (useLayerNorm)
    {
        MeanStddevNormalization(*cellScratchDecoder,
                                *cellScratch, nCell, nBatch, m_LayerNormEpsilon);
        VectorBatchVectorCwiseProduct(*cellLayerNormWeights,
                                      nCell, *cellScratchDecoder, nBatch, *cellScratch);
        VectorBatchVectorAdd(*cellBiasTensor,
                             nCell, *cellScratchDecoder, nBatch, *cellScratch);
    }

    VectorVectorCwiseProduct(*forgetGateScratchDecoder, *cellStateIn, nBatch * nCell, *cellStateOut);

    ActivationFunction armnnActivationFunc = ActivationFunction::Sigmoid;
    float a = 0;
    float b = 0;
    SetActivationParameters(m_Data.m_Parameters.m_ActivationFunc, armnnActivationFunc, a, b);

    if (m_Data.m_Parameters.m_ActivationFunc > 0)
    {
        Activation(*cellScratchDecoder, *cellScratch,
                   TensorInfo({nCell, nBatch}, outputType),
                   armnnActivationFunc, a, b);
    }
    if (useCifg)
    {
        Sub1Vector(*forgetGateScratchDecoder, nBatch * nCell, *forgetGateScratch);
        VectorVectorCwiseProductAccumulate(
            *cellScratchDecoder, *forgetGateScratchDecoder, nBatch * nCell, *cellStateOut);
    }
    else
    {
        VectorVectorCwiseProductAccumulate(
            *cellScratchDecoder, *inputGateScratchDecoder, nBatch * nCell, *cellStateOut);
    }
    if (m_Data.m_Parameters.m_ClippingThresCell > 0.0)
    {
        ClipVector(*cellStateOutDecoder, nBatch * nCell, m_Data.m_Parameters.m_ClippingThresCell, *cellStateOut);
    }

    // For each batch and cell: update the output gate.
    if (usePeephole)
    {
        VectorBatchVectorCwiseProductAccumulate(*cellToOutputWeightsTensor,
                                                nCell, *cellStateOutDecoder, nBatch, *outputGateScratch);
    }
    if (useLayerNorm)
    {
        MeanStddevNormalization(*outputGateScratchDecoder,
                                *outputGateScratch, nCell, nBatch, m_LayerNormEpsilon);
        VectorBatchVectorCwiseProduct(*outputLayerNormWeights,
                                      nCell, *outputGateScratchDecoder, nBatch, *outputGateScratch);
        VectorBatchVectorAdd(*outputGateBiasTensor,
                             nCell, *outputGateScratchDecoder, nBatch, *outputGateScratch);
    }
    Activation(*outputGateScratchDecoder, *outputGateScratch,
               TensorInfo({nCell, nBatch}, outputType),
               ActivationFunction::Sigmoid, 0, 0);

    if (m_Data.m_Parameters.m_ActivationFunc > 0)
    {
        Activation(*cellStateOutDecoder, *cellScratch,
                   TensorInfo({nCell, nBatch}, outputType),
                   armnnActivationFunc, a, b);
    }

    VectorVectorCwiseProduct(*outputGateScratchDecoder, *cellScratchDecoder, nBatch * nCell, *outputGateScratch);

    // For each batch: update the projection and output_state.
    if (m_Data.m_Parameters.m_ProjectionEnabled)
    {
        if (m_ProjectionBiasTensor)
        {
            VectorBatchVectorAssign(*projectionBiasTensor,
                                    nOutput, nBatch, *output);
        }
        MatrixBatchVectorMultiplyAccumulate(*projectionWeightsTensor,
                                            nOutput, nCell, *outputGateScratchDecoder, nBatch, *output);

        if (m_Data.m_Parameters.m_ClippingThresProj > 0.0)
        {
            ClipVector(*outputDecoder, nBatch * nOutput, m_Data.m_Parameters.m_ClippingThresProj, *output);
        }
    }
    else
    {
        CopyVector(*outputGateScratchDecoder, nBatch * nOutput, *output);
    }

    CopyVector(*outputDecoder, nBatch * nOutput, *outputStateOut);
}

} //namespace armnn
