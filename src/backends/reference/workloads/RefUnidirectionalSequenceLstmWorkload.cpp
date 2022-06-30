//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefUnidirectionalSequenceLstmWorkload.hpp"
#include "Activation.hpp"
#include "Encoders.hpp"
#include "Decoders.hpp"
#include "Lstm.hpp"
#include "LstmUtils.hpp"
#include "RefWorkloadUtils.hpp"

#include <armnnUtils/Permute.hpp>

namespace armnn
{

RefUnidirectionalSequenceLstmWorkload::RefUnidirectionalSequenceLstmWorkload(
    const UnidirectionalSequenceLstmQueueDescriptor& descriptor,
    const WorkloadInfo& info)
    : RefBaseWorkload<UnidirectionalSequenceLstmQueueDescriptor>(descriptor, info)
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
    , m_InputLayerNormWeights         (AssignScopedTensorHandle(descriptor.m_InputLayerNormWeights))
    , m_ForgetLayerNormWeights        (AssignScopedTensorHandle(descriptor.m_ForgetLayerNormWeights))
    , m_CellLayerNormWeights          (AssignScopedTensorHandle(descriptor.m_CellLayerNormWeights))
    , m_OutputLayerNormWeights        (AssignScopedTensorHandle(descriptor.m_OutputLayerNormWeights))
{}

void RefUnidirectionalSequenceLstmWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefUnidirectionalSequenceLstmWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefUnidirectionalSequenceLstmWorkload::Execute(std::vector<ITensorHandle*> inputs,
                                                    std::vector<ITensorHandle*> outputs) const
{
    TensorInfo inputInfo = GetTensorInfo(inputs[0]);
    const TensorInfo& outputStateInfo = GetTensorInfo(inputs[1]);
    const TensorInfo& cellStateInfo = GetTensorInfo(inputs[2]);
    TensorInfo outputStateOutInfo = GetTensorInfo(outputs[0]);
    TensorInfo cellStateOutInfo = GetTensorInfo(outputs[1]);
    TensorInfo outputInfo = GetTensorInfo(outputs[2]);
    TensorShape& inputShape = inputInfo.GetShape();
    TensorShape& outputShape= outputInfo.GetShape();
    auto inputTensor = reinterpret_cast<float*>(inputs[0]->Map());

    if (!m_Data.m_Parameters.m_TimeMajor)
    {
        // Permute to time major
        const PermutationVector& mappings = {1U, 0U, 2U};
        std::vector<float> inputValue(inputTensor, inputTensor + inputInfo.GetNumElements());
        inputShape = armnnUtils::Permuted(inputInfo.GetShape(), mappings);
        inputInfo.SetShape(inputShape);
        armnnUtils::Permute(inputShape, mappings,  inputValue.data(), inputTensor, sizeof(float));

        outputShape = armnnUtils::Permuted(outputInfo.GetShape(), mappings);
        outputInfo.SetShape(outputShape);
    }
    unsigned int maxTime = inputShape[0];
    unsigned int batchSize = inputShape[1];
    unsigned int outputSize = outputShape[2];
    unsigned int inputSize = inputShape[2];

    TensorInfo scratchInfo = outputInfo;
    scratchInfo.SetShape({batchSize, cellStateInfo.GetShape()[1]});

    std::vector<float> inputGateScratchBuffer;
    std::vector<float> cellScratchBuffer(scratchInfo.GetNumElements(), 0.);
    std::vector<float> forgetGateScratchBuffer(scratchInfo.GetNumElements(), 0.);
    std::vector<float> outputGateScratchBuffer(scratchInfo.GetNumElements(), 0.);

    std::vector<float> outputStateOutBuffer(outputStateInfo.GetNumElements(), 0.);
    std::vector<float> cellStateOutBuffer(cellStateInfo.GetNumElements(), 0.);

    void* outputStateOutData = outputStateOutBuffer.data();
    void* cellStateOutData = cellStateOutBuffer.data();

    std::unique_ptr<Encoder<float>> inputGateScratch;
    std::unique_ptr<Encoder<float>> cellScratch = MakeEncoder<float>(scratchInfo, cellScratchBuffer.data());
    std::unique_ptr<Encoder<float>> forgetGateScratch = MakeEncoder<float>(scratchInfo, forgetGateScratchBuffer.data());
    std::unique_ptr<Encoder<float>> outputGateScratch = MakeEncoder<float>(scratchInfo, outputGateScratchBuffer.data());

    std::unique_ptr<Decoder<float>> inputGateScratchDecoder;
    std::unique_ptr<Decoder<float>> cellScratchDecoder = MakeDecoder<float>(scratchInfo, cellScratchBuffer.data());
    std::unique_ptr<Decoder<float>> forgetGateScratchDecoder = MakeDecoder<float>(scratchInfo,
                                                                                  forgetGateScratchBuffer.data());
    std::unique_ptr<Decoder<float>> outputGateScratchDecoder = MakeDecoder<float>(scratchInfo,
                                                                                  outputGateScratchBuffer.data());

    const bool useCifg      = m_Data.m_Parameters.m_CifgEnabled;
    const bool usePeephole  = m_Data.m_Parameters.m_PeepholeEnabled;
    const bool useLayerNorm = m_Data.m_Parameters.m_LayerNormEnabled;

    if (!useCifg)
    {
        inputGateScratchBuffer.resize(scratchInfo.GetNumElements(), 0.);
        inputGateScratch = MakeEncoder<float>(scratchInfo, inputGateScratchBuffer.data());
        inputGateScratchDecoder = MakeDecoder<float>(scratchInfo, inputGateScratchBuffer.data());
    }

    std::unique_ptr<Encoder<float>> outputStateOut = MakeEncoder<float>(outputStateInfo, outputStateOutData);
    std::unique_ptr<Encoder<float>> cellStateOut   = MakeEncoder<float>(cellStateInfo, cellStateOutData);
    std::unique_ptr<Decoder<float>> cellStateOutDecoder = MakeDecoder<float>(cellStateInfo, cellStateOutData);

    TensorInfo lstmInputInfo = inputInfo;
    TensorShape batchInputShape = TensorShape({batchSize, inputSize});
    lstmInputInfo.SetShape(batchInputShape);

    TensorInfo lstmOutputInfo = outputInfo;
    lstmOutputInfo.SetShape({batchSize, outputSize});

    const TensorShape& inputToOutputWeightsShape = m_InputToOutputWeightsTensor->GetShape();
    const TensorShape& recurrentToOutputWeightsShape = m_RecurrentToOutputWeightsTensor->GetShape();
    unsigned int nOutput = recurrentToOutputWeightsShape[1];
    auto outputStateInData = inputs[1]->Map();
    std::unique_ptr<Decoder<float>> outputStateIn = MakeDecoder<float>(outputStateInfo, outputStateInData);

    auto cellStateInData = inputs[2]->Map();
    std::unique_ptr<Decoder<float>> cellStateIn = MakeDecoder<float>(cellStateInfo, cellStateInData);

    auto currentInputData = reinterpret_cast<float*>(inputs[0]->Map());
    std::unique_ptr<Decoder<float>> inputData = MakeDecoder<float>(lstmInputInfo, currentInputData);
    auto currentOutputData = reinterpret_cast<float*>(outputs[2]->Map());
    std::unique_ptr<Encoder<float>> output = MakeEncoder<float>(lstmOutputInfo, currentOutputData);
    std::unique_ptr<Decoder<float>> outputDecoder = MakeDecoder<float>(lstmOutputInfo, currentOutputData);

    std::unique_ptr<Decoder<float>> inputToInputWeightsTensor;
    std::unique_ptr<Decoder<float>> inputToForgetWeightsTensor = MakeDecoder<float>(
        m_InputToForgetWeightsTensor->GetTensorInfo(), m_InputToForgetWeightsTensor->GetConstTensor<void>());
    std::unique_ptr<Decoder<float>> inputToCellWeightsTensor = MakeDecoder<float>(
        m_InputToCellWeightsTensor->GetTensorInfo(), m_InputToCellWeightsTensor->GetConstTensor<void>());
    std::unique_ptr<Decoder<float>> inputToOutputWeightsTensor = MakeDecoder<float>(
        m_InputToOutputWeightsTensor->GetTensorInfo(), m_InputToOutputWeightsTensor->GetConstTensor<void>());

    std::unique_ptr<Decoder<float>> recurrentToInputWeightsTensor;
    std::unique_ptr<Decoder<float>> recurrentToForgetWeightsTensor = MakeDecoder<float>(
        m_RecurrentToForgetWeightsTensor->GetTensorInfo(), m_RecurrentToForgetWeightsTensor->GetConstTensor<void>());
    std::unique_ptr<Decoder<float>> recurrentToCellWeightsTensor = MakeDecoder<float>(
        m_RecurrentToCellWeightsTensor->GetTensorInfo(), m_RecurrentToCellWeightsTensor->GetConstTensor<void>());
    std::unique_ptr<Decoder<float>> recurrentToOutputWeightsTensor = MakeDecoder<float>(
        m_RecurrentToOutputWeightsTensor->GetTensorInfo(), m_RecurrentToOutputWeightsTensor->GetConstTensor<void>());

    std::unique_ptr<Decoder<float>> inputGateBiasTensor;
    std::unique_ptr<Decoder<float>> forgetGateBiasTensor = MakeDecoder<float>(
        m_ForgetGateBiasTensor->GetTensorInfo(), m_ForgetGateBiasTensor->GetConstTensor<void>());
    std::unique_ptr<Decoder<float>> cellBiasTensor = MakeDecoder<float>(
        m_CellBiasTensor->GetTensorInfo(), m_CellBiasTensor->GetConstTensor<void>());
    std::unique_ptr<Decoder<float>> outputGateBiasTensor = MakeDecoder<float>(
        m_OutputGateBiasTensor->GetTensorInfo(), m_OutputGateBiasTensor->GetConstTensor<void>());

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
                    m_InputLayerNormWeights->GetTensorInfo(), m_InputLayerNormWeights->GetConstTensor<void>());
        }
        forgetLayerNormWeights = MakeDecoder<float>(
                m_ForgetLayerNormWeights->GetTensorInfo(), m_ForgetLayerNormWeights->GetConstTensor<void>());
        cellLayerNormWeights = MakeDecoder<float>(
                m_CellLayerNormWeights->GetTensorInfo(), m_CellLayerNormWeights->GetConstTensor<void>());
        outputLayerNormWeights = MakeDecoder<float>(
                m_OutputLayerNormWeights->GetTensorInfo(), m_OutputLayerNormWeights->GetConstTensor<void>());
    }

    if (!useCifg)
    {
        inputToInputWeightsTensor = MakeDecoder<float>(
            m_InputToInputWeightsTensor->GetTensorInfo(), m_InputToInputWeightsTensor->GetConstTensor<void>());
        inputGateBiasTensor = MakeDecoder<float>(
            m_InputGateBiasTensor->GetTensorInfo(), m_InputGateBiasTensor->GetConstTensor<void>());
        recurrentToInputWeightsTensor = MakeDecoder<float>(
            m_RecurrentToInputWeightsTensor->GetTensorInfo(), m_RecurrentToInputWeightsTensor->GetConstTensor<void>());
    }

    if (usePeephole)
    {
        cellToForgetWeightsTensor = MakeDecoder<float>(
            m_CellToForgetWeightsTensor->GetTensorInfo(), m_CellToForgetWeightsTensor->GetConstTensor<void>());
        cellToOutputWeightsTensor = MakeDecoder<float>(
            m_CellToOutputWeightsTensor->GetTensorInfo(), m_CellToOutputWeightsTensor->GetConstTensor<void>());
    }

    if (!useCifg && usePeephole)
    {
        cellToInputWeightsTensor = MakeDecoder<float>(
            m_CellToInputWeightsTensor->GetTensorInfo(), m_CellToInputWeightsTensor->GetConstTensor<void>());
    }

    if (m_Data.m_Parameters.m_ProjectionEnabled)
    {
        projectionWeightsTensor = MakeDecoder<float>(
            m_ProjectionWeightsTensor->GetTensorInfo(), m_ProjectionWeightsTensor->GetConstTensor<void>());
        if (m_ProjectionBiasTensor)
        {
            projectionBiasTensor = MakeDecoder<float>(
                m_ProjectionBiasTensor->GetTensorInfo(), m_ProjectionBiasTensor->GetConstTensor<void>());
        }
    }

    unsigned int batchInputSize = batchSize * inputSize;
    unsigned int batchOutputSize = batchSize * nOutput;

    for (unsigned int t = 0; t < maxTime; ++t)
    {
        LstmImpl(m_Data.m_Parameters,
                 lstmInputInfo,
                 lstmOutputInfo,
                 inputToOutputWeightsShape,
                 recurrentToOutputWeightsShape,
                 inputData,
                 outputStateIn,
                 cellStateIn,
                 outputStateOut,
                 cellStateOut,
                 output,
                 cellStateOutDecoder,
                 outputDecoder,
                 inputToInputWeightsTensor,
                 inputToForgetWeightsTensor,
                 inputToCellWeightsTensor,
                 inputToOutputWeightsTensor,
                 recurrentToInputWeightsTensor,
                 recurrentToForgetWeightsTensor,
                 recurrentToCellWeightsTensor,
                 recurrentToOutputWeightsTensor,
                 cellToInputWeightsTensor,
                 cellToForgetWeightsTensor,
                 cellToOutputWeightsTensor,
                 inputGateBiasTensor,
                 forgetGateBiasTensor,
                 cellBiasTensor,
                 outputGateBiasTensor,
                 projectionWeightsTensor,
                 projectionBiasTensor,
                 inputLayerNormWeights,
                 forgetLayerNormWeights,
                 cellLayerNormWeights,
                 outputLayerNormWeights,
                 inputGateScratch,
                 cellScratch,
                 forgetGateScratch,
                 outputGateScratch,
                 inputGateScratchDecoder,
                 cellScratchDecoder,
                 forgetGateScratchDecoder,
                 outputGateScratchDecoder,
                 m_LayerNormEpsilon);

        currentInputData += batchInputSize;
        inputData = MakeDecoder<float>(lstmInputInfo, currentInputData);
        currentOutputData += batchOutputSize;
        output = MakeEncoder<float>(lstmOutputInfo, currentOutputData);
        outputDecoder = MakeDecoder<float>(lstmOutputInfo, currentOutputData);

        // Assign output state out to the next output state in
        outputStateIn = MakeDecoder<float>(outputStateInfo, outputStateOutData);

        // Assign cell state out to the next cell state in
        cellStateIn = MakeDecoder<float>(cellStateInfo, cellStateOutData);
    }

    if (!m_Data.m_Parameters.m_TimeMajor)
    {
        // Permute Output back to batch major
        const PermutationVector& mappings = {1U, 0U, 2U};
        auto outputData = reinterpret_cast<float*>(outputs[2]->Map());
        std::vector<float> outputValue(outputData, outputData + outputInfo.GetNumElements());
        outputShape = armnnUtils::Permuted(outputInfo.GetShape(), mappings);
        outputInfo.SetShape(outputShape);
        armnnUtils::Permute(outputShape, mappings, outputValue.data(), outputData, sizeof(float));
    }
}

} //namespace armnn
