//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefLstmFloat32Workload.hpp"
#include "RefWorkloadUtils.hpp"
#include "Activation.hpp"

namespace
{

// Helper functions ported from the Android code base
// Refer to: android/external/tensorflow/tensorflow/contrib/lite/kernels/internal/reference/portable_tensor_utils.cc

void MatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                         uint32_t mRows,
                                         uint32_t mCols,
                                         const float* vector,
                                         uint32_t nBatch,
                                         float* outResult,
                                         int resultStride = 1)
{
    float* resultInBatch = outResult;
    for (uint32_t b = 0; b < nBatch; b++)
    {
        const float* matrixPtr = matrix;
        for (uint32_t r = 0; r < mRows; r++)
        {
            const float* vectorInBatch = vector + b * mCols;
            for (uint32_t c = 0; c < mCols; c++)
            {
                *resultInBatch += *matrixPtr++ * *vectorInBatch++;
            }
            resultInBatch += resultStride;
        }
    }
}

void VectorBatchVectorAssign(const float* vector,
                             uint32_t vSize,
                             uint32_t nBatch,
                             float* outBatchVector)
{
    for (uint32_t b = 0; b < nBatch; b++)
    {
        memcpy(outBatchVector + b * vSize, vector, vSize * sizeof(float));
    }
}

void VectorBatchVectorCwiseProductAccumulate(const float* vector,
                                             uint32_t vSize,
                                             const float* batchVector,
                                             uint32_t nBatch,
                                             float* outResult)
{
    for (uint32_t b = 0; b < nBatch; b++)
    {
        for (uint32_t v = 0; v < vSize; v++)
        {
            *outResult++ += vector[v] * *batchVector++;
        }
    }
}

void Sub1Vector(const float* vector,
                uint32_t vSize,
                float* result)
{
    for (uint32_t v = 0; v < vSize; v++)
    {
        *result++ = 1.0f - *vector++;
    }
}

void VectorVectorCwiseProduct(const float* vector1,
                              const float* vector2,
                              uint32_t vSize,
                              float* outResult)
{
    for (uint32_t v = 0; v < vSize; v++)
    {
        *outResult++ = *vector1++ * *vector2++;
    }
}

void VectorVectorCwiseProductAccumulate(const float* vector1,
                                        const float* vector2,
                                        uint32_t vSize,
                                        float* outResult)
{
    for (uint32_t v = 0; v < vSize; v++)
    {
        *outResult++ += *vector1++ * *vector2++;
    }
}

float Clip(float f,
           float absLimit)
{
    float result = (absLimit < f) ? absLimit : f;
    result = (-absLimit > result) ? -absLimit : result;
    return result;
}

void ClipVector(const float* vector,
                uint32_t vSize,
                float absLimit,
                float* outResult)
{
    for (uint32_t v = 0; v < vSize; v++)
    {
        *outResult++ = Clip(*vector++, absLimit);
    }
}

void CopyVector(const float* vector,
                uint32_t vSize,
                float* outResult)
{
    memcpy(outResult, vector, vSize * sizeof(float));
}

void SetActivationParameters(uint32_t activation,
                             armnn::ActivationFunction& outArmnnActivation,
                             float& outA,
                             float& outB)
{
    switch (activation)
    {
    case 0: // None
        outA = 0;
        outB = 0;
        return;

    case 1: // Relu
        outArmnnActivation = armnn::ActivationFunction::ReLu;
        outA = 0;
        outB = 0;
        return;

    case 3: // Relu6
        outArmnnActivation = armnn::ActivationFunction::BoundedReLu;
        outA = 6;
        outB = 0;
        return;

    case 4: // Tanh
        outArmnnActivation = armnn::ActivationFunction::TanH;
        outA = 1;
        outB = 1;
        return;

    case 6: // Sigmoid
        outArmnnActivation = armnn::ActivationFunction::Sigmoid;
        outA = 0;
        outB = 0;
        return;

    default:
        throw armnn::Exception("Unsupported activation function: " + std::to_string(activation));
    }
}

std::unique_ptr<armnn::ScopedCpuTensorHandle> AssignScopedCpuTensorHandle(const armnn::ConstCpuTensorHandle* ptr)
{
    if (!ptr)
    {
        return nullptr;
    }

    return std::make_unique<armnn::ScopedCpuTensorHandle>(*ptr);
}

} // anonymous namespace

namespace armnn
{

RefLstmFloat32Workload::RefLstmFloat32Workload(const LstmQueueDescriptor &descriptor, const WorkloadInfo &info)
    : Float32Workload<LstmQueueDescriptor>(descriptor, info)
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
{}

void RefLstmFloat32Workload::Execute() const
{
    // This is a porting of the LSTM::Eval() method in the Android code base
    // Refer to: android/frameworks/ml/nn/common/operations/LSTM.cpp

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorShape& inputShape = inputInfo.GetShape();

    float* scratchBuffer  = GetOutputTensorDataFloat(0, m_Data);
    float* outputStateOut = GetOutputTensorDataFloat(1, m_Data);
    float* cellStateOut   = GetOutputTensorDataFloat(2, m_Data);
    float* output         = GetOutputTensorDataFloat(3, m_Data);

    const float* inputData     = GetInputTensorDataFloat(0, m_Data);
    const float* outputStateIn = GetInputTensorDataFloat(1, m_Data);
    const float* cellStateIn   = GetInputTensorDataFloat(2, m_Data);

    const uint32_t nBatch = inputShape[0];
    const uint32_t nInput = inputShape[1];

    const uint32_t nCell   = m_InputToOutputWeightsTensor->GetShape()[0];
    const uint32_t nOutput = m_RecurrentToOutputWeightsTensor->GetShape()[1];

    const bool useCifg     = m_Data.m_Parameters.m_CifgEnabled;
    const bool usePeephole = m_Data.m_Parameters.m_PeepholeEnabled;

    // Index the scratch buffers pointers to the global scratch buffer.
    float* inputGateScratch  = nullptr;
    float* cellScratch       = nullptr;
    float* forgetGateScratch = nullptr;
    float* outputGateScratch = nullptr;

    if (useCifg)
    {
        cellScratch       = scratchBuffer + 0 * nCell * nBatch;
        forgetGateScratch = scratchBuffer + 1 * nCell * nBatch;
        outputGateScratch = scratchBuffer + 2 * nCell * nBatch;
    }
    else
    {
        inputGateScratch  = scratchBuffer + 0 * nCell * nBatch;
        cellScratch       = scratchBuffer + 1 * nCell * nBatch;
        forgetGateScratch = scratchBuffer + 2 * nCell * nBatch;
        outputGateScratch = scratchBuffer + 3 * nCell * nBatch;
    }

    // Initialize scratch buffers with bias.
    if (!useCifg)
    {
        VectorBatchVectorAssign(m_InputGateBiasTensor->GetTensor<float>(),
                                nCell, nBatch, inputGateScratch);
    }
    VectorBatchVectorAssign(m_ForgetGateBiasTensor->GetTensor<float>(),
                            nCell, nBatch, forgetGateScratch);
    VectorBatchVectorAssign(m_CellBiasTensor->GetTensor<float>(),
                            nCell, nBatch, cellScratch);
    VectorBatchVectorAssign(m_OutputGateBiasTensor->GetTensor<float>(),
                            nCell, nBatch, outputGateScratch);

    // For each batch and cell: compute input_weight * input.
    if (!useCifg)
    {
        MatrixBatchVectorMultiplyAccumulate(m_InputToInputWeightsTensor->GetTensor<float>(),
                                            nCell, nInput, inputData, nBatch, inputGateScratch);
    }
    MatrixBatchVectorMultiplyAccumulate(m_InputToForgetWeightsTensor->GetTensor<float>(),
                                        nCell, nInput, inputData, nBatch, forgetGateScratch);
    MatrixBatchVectorMultiplyAccumulate(m_InputToCellWeightsTensor->GetTensor<float>(),
                                        nCell, nInput, inputData, nBatch, cellScratch);
    MatrixBatchVectorMultiplyAccumulate(m_InputToOutputWeightsTensor->GetTensor<float>(),
                                        nCell, nInput, inputData, nBatch, outputGateScratch);

    // For each batch and cell: compute recurrent_weight * output_state.
    if (!useCifg)
    {
        MatrixBatchVectorMultiplyAccumulate(m_RecurrentToInputWeightsTensor->GetTensor<float>(),
                                            nCell, nOutput, outputStateIn, nBatch, inputGateScratch);
    }
    MatrixBatchVectorMultiplyAccumulate(m_RecurrentToForgetWeightsTensor->GetTensor<float>(),
                                        nCell, nOutput, outputStateIn, nBatch, forgetGateScratch);
    MatrixBatchVectorMultiplyAccumulate(m_RecurrentToCellWeightsTensor->GetTensor<float>(),
                                        nCell, nOutput, outputStateIn, nBatch, cellScratch);
    MatrixBatchVectorMultiplyAccumulate(m_RecurrentToOutputWeightsTensor->GetTensor<float>(),
                                        nCell, nOutput, outputStateIn, nBatch, outputGateScratch);

    // For each batch and cell: update input gate.
    if (!useCifg)
    {
        if (usePeephole)
        {
            VectorBatchVectorCwiseProductAccumulate(m_CellToInputWeightsTensor->GetTensor<float>(),
                                                    nCell, cellStateIn, nBatch, inputGateScratch);
        }
        Activation(inputGateScratch, inputGateScratch,
                   TensorInfo({nCell, nBatch}, DataType::Float32),
                   ActivationFunction::Sigmoid, 0, 0);
    }

    // For each batch and cell: update forget gate.
    if (usePeephole)
    {
        VectorBatchVectorCwiseProductAccumulate(m_CellToForgetWeightsTensor->GetTensor<float>(), nCell,
                                                cellStateIn, nBatch, forgetGateScratch);
    }
    Activation(forgetGateScratch, forgetGateScratch,
               TensorInfo({nCell, nBatch}, DataType::Float32),
               ActivationFunction::Sigmoid, 0, 0);

    // For each batch and cell: update the cell.
    VectorVectorCwiseProduct(forgetGateScratch, cellStateIn, nBatch * nCell, cellStateOut);

    ActivationFunction armnnActivationFunc = ActivationFunction::Sigmoid;
    float a = 0;
    float b = 0;
    SetActivationParameters(m_Data.m_Parameters.m_ActivationFunc, armnnActivationFunc, a, b);

    if (m_Data.m_Parameters.m_ActivationFunc > 0)
    {
        Activation(cellScratch, cellScratch,
                   TensorInfo({nCell, nBatch}, DataType::Float32),
                   armnnActivationFunc, a, b);
    }
    if (useCifg)
    {
        Sub1Vector(forgetGateScratch, nBatch * nCell, forgetGateScratch);
        VectorVectorCwiseProductAccumulate(cellScratch, forgetGateScratch, nBatch * nCell, cellStateOut);
    }
    else
    {
        VectorVectorCwiseProductAccumulate(cellScratch, inputGateScratch, nBatch * nCell, cellStateOut);
    }
    if (m_Data.m_Parameters.m_ClippingThresCell > 0.0)
    {
        ClipVector(cellStateOut, nBatch * nCell, m_Data.m_Parameters.m_ClippingThresCell, cellStateOut);
    }

    // For each batch and cell: update the output gate.
    if (usePeephole)
    {
        VectorBatchVectorCwiseProductAccumulate(m_CellToOutputWeightsTensor->GetTensor<float>(),
                                                nCell, cellStateOut, nBatch, outputGateScratch);
    }
    Activation(outputGateScratch, outputGateScratch,
               TensorInfo({nCell, nBatch}, DataType::Float32),
               ActivationFunction::Sigmoid, 0, 0);

    if (m_Data.m_Parameters.m_ActivationFunc > 0)
    {
        Activation(cellStateOut, cellScratch,
                   TensorInfo({nCell, nBatch}, DataType::Float32),
                   armnnActivationFunc, a, b);
    }
    VectorVectorCwiseProduct(outputGateScratch, cellScratch, nBatch * nCell, outputGateScratch);

    // For each batch: update the projection and output_state.
    if (m_Data.m_Parameters.m_ProjectionEnabled)
    {
        if (m_ProjectionBiasTensor)
        {
            VectorBatchVectorAssign(m_ProjectionBiasTensor->GetTensor<float>(),
                                    nOutput, nBatch, output);
        }
        MatrixBatchVectorMultiplyAccumulate(m_ProjectionWeightsTensor->GetTensor<float>(),
                                            nOutput, nCell, outputGateScratch, nBatch, output);

        if (m_Data.m_Parameters.m_ClippingThresProj > 0.0)
        {
            ClipVector(output, nBatch * nOutput, m_Data.m_Parameters.m_ClippingThresProj, output);
        }
    }
    else
    {
        CopyVector(outputGateScratch, nBatch * nOutput, output);
    }

    CopyVector(output, nBatch * nOutput, outputStateOut);
}

} //namespace armnn
