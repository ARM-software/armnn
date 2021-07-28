//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Activation.hpp"
#include "Lstm.hpp"
#include "LstmUtils.hpp"

namespace armnn
{

void LstmImpl(const LstmDescriptor& descriptor,
              const TensorInfo& inputInfo,
              const TensorInfo& outputInfo,
              const TensorShape& inputToOutputWeightsShape,
              const TensorShape& recurrentToOutputWeightsShape,
              std::unique_ptr<Decoder<float>>& inputData,
              std::unique_ptr<Decoder<float>>& outputStateIn,
              std::unique_ptr<Decoder<float>>& cellStateIn,
              std::unique_ptr<Encoder<float>>& outputStateOut,
              std::unique_ptr<Encoder<float>>& cellStateOut,
              std::unique_ptr<Encoder<float>>& output,
              std::unique_ptr<Decoder<float>>& cellStateOutDecoder,
              std::unique_ptr<Decoder<float>>& outputDecoder,
              std::unique_ptr<Decoder<float>>& inputToInputWeightsTensor,
              std::unique_ptr<Decoder<float>>& inputToForgetWeightsTensor,
              std::unique_ptr<Decoder<float>>& inputToCellWeightsTensor,
              std::unique_ptr<Decoder<float>>& inputToOutputWeightsTensor,
              std::unique_ptr<Decoder<float>>& recurrentToInputWeightsTensor,
              std::unique_ptr<Decoder<float>>& recurrentToForgetWeightsTensor,
              std::unique_ptr<Decoder<float>>& recurrentToCellWeightsTensor,
              std::unique_ptr<Decoder<float>>& recurrentToOutputWeightsTensor,
              std::unique_ptr<Decoder<float>>& cellToInputWeightsTensor,
              std::unique_ptr<Decoder<float>>& cellToForgetWeightsTensor,
              std::unique_ptr<Decoder<float>>& cellToOutputWeightsTensor,
              std::unique_ptr<Decoder<float>>& inputGateBiasTensor,
              std::unique_ptr<Decoder<float>>& forgetGateBiasTensor,
              std::unique_ptr<Decoder<float>>& cellBiasTensor,
              std::unique_ptr<Decoder<float>>& outputGateBiasTensor,
              std::unique_ptr<Decoder<float>>& projectionWeightsTensor,
              std::unique_ptr<Decoder<float>>& projectionBiasTensor,
              std::unique_ptr<Decoder<float>>& inputLayerNormWeights,
              std::unique_ptr<Decoder<float>>& forgetLayerNormWeights,
              std::unique_ptr<Decoder<float>>& cellLayerNormWeights,
              std::unique_ptr<Decoder<float>>& outputLayerNormWeights,
              std::unique_ptr<Encoder<float>>& inputGateScratch,
              std::unique_ptr<Encoder<float>>& cellScratch,
              std::unique_ptr<Encoder<float>>& forgetGateScratch,
              std::unique_ptr<Encoder<float>>& outputGateScratch,
              std::unique_ptr<Decoder<float>>& inputGateScratchDecoder,
              std::unique_ptr<Decoder<float>>& cellScratchDecoder,
              std::unique_ptr<Decoder<float>>& forgetGateScratchDecoder,
              std::unique_ptr<Decoder<float>>& outputGateScratchDecoder,
              float layerNormEpsilon)
{
    // This is a porting of the LSTM::Eval() method in the Android code base
    // Refer to: android/frameworks/ml/nn/common/operations/LSTM.cpp

    const TensorShape& inputShape = inputInfo.GetShape();
    const DataType& outputType = outputInfo.GetDataType();

    const uint32_t nBatch = inputShape[0];
    const uint32_t nInput = inputShape[1];

    const uint32_t nCell   = inputToOutputWeightsShape[0];
    const uint32_t nOutput = recurrentToOutputWeightsShape[1];

    const bool useCifg      = descriptor.m_CifgEnabled;
    const bool usePeephole  = descriptor.m_PeepholeEnabled;
    const bool useLayerNorm = descriptor.m_LayerNormEnabled;

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
                                    *inputGateScratch, nCell, nBatch, layerNormEpsilon);
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
                                *forgetGateScratch, nCell, nBatch, layerNormEpsilon);
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
                                *cellScratch, nCell, nBatch, layerNormEpsilon);
        VectorBatchVectorCwiseProduct(*cellLayerNormWeights,
                                      nCell, *cellScratchDecoder, nBatch, *cellScratch);
        VectorBatchVectorAdd(*cellBiasTensor,
                             nCell, *cellScratchDecoder, nBatch, *cellScratch);
    }

    VectorVectorCwiseProduct(*forgetGateScratchDecoder, *cellStateIn, nBatch * nCell, *cellStateOut);

    ActivationFunction armnnActivationFunc = ActivationFunction::Sigmoid;
    float a = 0;
    float b = 0;
    SetActivationParameters(descriptor.m_ActivationFunc, armnnActivationFunc, a, b);

    if (descriptor.m_ActivationFunc > 0)
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
    if (descriptor.m_ClippingThresCell > 0.0)
    {
        ClipVector(*cellStateOutDecoder, nBatch * nCell, descriptor.m_ClippingThresCell, *cellStateOut);
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
                                *outputGateScratch, nCell, nBatch, layerNormEpsilon);
        VectorBatchVectorCwiseProduct(*outputLayerNormWeights,
                                      nCell, *outputGateScratchDecoder, nBatch, *outputGateScratch);
        VectorBatchVectorAdd(*outputGateBiasTensor,
                             nCell, *outputGateScratchDecoder, nBatch, *outputGateScratch);
    }
    Activation(*outputGateScratchDecoder, *outputGateScratch,
               TensorInfo({nCell, nBatch}, outputType),
               ActivationFunction::Sigmoid, 0, 0);

    if (descriptor.m_ActivationFunc > 0)
    {
        Activation(*cellStateOutDecoder, *cellScratch,
                   TensorInfo({nCell, nBatch}, outputType),
                   armnnActivationFunc, a, b);
    }

    VectorVectorCwiseProduct(*outputGateScratchDecoder, *cellScratchDecoder, nBatch * nCell, *outputGateScratch);

    // For each batch: update the projection and output_state.
    if (descriptor.m_ProjectionEnabled)
    {
        if (projectionBiasTensor)
        {
            VectorBatchVectorAssign(*projectionBiasTensor,
                                    nOutput, nBatch, *output);
        }
        MatrixBatchVectorMultiplyAccumulate(*projectionWeightsTensor,
                                            nOutput, nCell, *outputGateScratchDecoder, nBatch, *output);

        if (descriptor.m_ClippingThresProj > 0.0)
        {
            ClipVector(*outputDecoder, nBatch * nOutput, descriptor.m_ClippingThresProj, *output);
        }
    }
    else
    {
        CopyVector(*outputGateScratchDecoder, nBatch * nOutput, *output);
    }

    CopyVector(*outputDecoder, nBatch * nOutput, *outputStateOut);
}

} //namespace armnn
