//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

//#pragma once

#include "LstmUtils.hpp"
#include "BaseIterator.hpp"
#include <armnn/backends/TensorHandle.hpp>


// Helper functions ported from the Android code base
// Refer to: android/external/tensorflow/tensorflow/contrib/lite/kernels/internal/reference/portable_tensor_utils.cc

void VectorBatchVectorAdd(armnn::Decoder<float>& vector,
                          uint32_t vSize,
                          armnn::Decoder<float>& batchVector,
                          uint32_t nBatch,
                          armnn::Encoder<float>& outResult )
{
    for (uint32_t b = 0; b < nBatch; b++)
    {
        for (uint32_t v = 0; v < vSize; v++)
        {
            outResult.Set(batchVector.Get() + vector.Get());
            ++outResult;
            ++vector;
            ++batchVector;
        }
        vector -= vSize;
    }
    batchVector -= vSize * nBatch;
    outResult -= vSize * nBatch;
}


// Layer norm for each batch.
// normalization_epsilon is added to avoid divergence.
void MeanStddevNormalization(armnn::Decoder<float>& input_vector,
                             armnn::Encoder<float>& output_vector,
                             uint32_t v_size,
                             uint32_t n_batch,
                             float normalization_epsilon)
{
    for (uint32_t batch = 0; batch < n_batch; ++batch) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < v_size; ++i) {
            sum += input_vector.Get();
            sum_sq += input_vector.Get() * input_vector.Get();
            ++input_vector;
        }
        input_vector -= v_size;

        const float mean = sum / static_cast<float>(v_size);
        float stddev_inv = 0.0f;
        const float variance = sum_sq / static_cast<float>(v_size) - mean * mean;
        if (variance == 0) {
            stddev_inv = 1.0f / std::sqrt(normalization_epsilon);
        } else {
            stddev_inv = 1.0f / std::sqrt(variance);
        }

        for (uint32_t i = 0; i < v_size; ++i) {
            output_vector.Set((input_vector.Get() - mean) * stddev_inv);
            ++output_vector;
            ++input_vector;
        }
        // Don't reset iterator to handle next batch
    }
    output_vector -= v_size * n_batch;
    input_vector -= v_size * n_batch;
}

void ZeroVector(armnn::Encoder<float>& vector,
                uint32_t vSize)
{
    for (uint32_t v = 0; v < vSize; v++)
    {
        vector.Set(0.0f);
        ++vector;
    }
    vector -= vSize;
}

void MatrixBatchVectorMultiplyAccumulate(armnn::Decoder<float>& matrix,
                                         uint32_t mRows,
                                         uint32_t mCols,
                                         armnn::Decoder<float>& vector,
                                         uint32_t nBatch,
                                         armnn::Encoder<float>& outResult)
{
    for (uint32_t b = 0; b < nBatch; b++)
    {
        for (uint32_t r = 0; r < mRows; r++)
        {
            vector += b * mCols;
            for (uint32_t c = 0; c < mCols; c++)
            {
                outResult.Set(outResult.Get() + matrix.Get() * vector.Get());
                ++matrix;
                ++vector;
            }
            outResult += 1;
            vector -= (b+1) * mCols;
        }
        matrix -= (mRows * mCols);
    }
    outResult -= (mRows * nBatch);
}

void VectorBatchVectorAssign(armnn::Decoder<float>& vector,
                             uint32_t vSize,
                             uint32_t nBatch,
                             armnn::Encoder<float>& outBatchVector)
{
    for (uint32_t b = 0; b < nBatch; b++)
    {
        for (uint32_t v = 0; v < vSize; v++)
        {
            outBatchVector.Set(vector.Get());
            ++outBatchVector;
            ++vector;
        }
        vector -= vSize;
    }
    outBatchVector -= (nBatch * vSize);
}

void VectorBatchVectorCwiseProductAccumulate(armnn::Decoder<float>& vector,
                                             uint32_t vSize,
                                             armnn::Decoder<float>& batchVector,
                                             uint32_t nBatch,
                                             armnn::Encoder<float>& outResult)
{
    for (uint32_t b = 0; b < nBatch; b++)
    {
        for (uint32_t v = 0; v < vSize; v++)
        {
            outResult.Set(outResult.Get() + vector.Get() * batchVector.Get());
            ++outResult;
            ++vector;
            ++batchVector;
        }
        vector -= vSize;
    }
    batchVector -= vSize * nBatch;
    outResult -= vSize * nBatch;
}

void VectorBatchVectorCwiseProduct(armnn::Decoder<float>& vector,
                                   uint32_t vSize,
                                   armnn::Decoder<float>& batchVector,
                                   uint32_t nBatch,
                                   armnn::Encoder<float>& outResult)
{
    for (uint32_t b = 0; b < nBatch; b++)
    {
        for (uint32_t v = 0; v < vSize; v++)
        {
            outResult.Set(vector.Get() * batchVector.Get());
            ++outResult;
            ++vector;
            ++batchVector;
        }
        vector -= vSize;
    }
    batchVector -= vSize * nBatch;
    outResult -= vSize * nBatch;
}

void Sub1Vector(armnn::Decoder<float>& vector,
                uint32_t vSize,
                armnn::Encoder<float>& result)
{
    for (uint32_t v = 0; v < vSize; v++)
    {
        result.Set(1.0f - vector.Get());
        ++vector;
        ++result;
    }
    vector -= vSize;
    result -= vSize;
}

void VectorVectorCwiseProduct(armnn::Decoder<float>& vector1,
                              armnn::Decoder<float>& vector2,
                              uint32_t vSize,
                              armnn::Encoder<float>& outResult)
{
    for (uint32_t v = 0; v < vSize; v++)
    {
        outResult.Set(vector1.Get() * vector2.Get());
        ++outResult;
        ++vector1;
        ++vector2;
    }
    outResult -= vSize;
    vector1 -= vSize;
    vector2 -= vSize;
}

void VectorVectorCwiseProductAccumulate(armnn::Decoder<float>& vector1,
                                        armnn::Decoder<float>& vector2,
                                        uint32_t vSize,
                                        armnn::Encoder<float>& outResult)
{
    for (uint32_t v = 0; v < vSize; v++)
    {
        outResult.Set(outResult.Get() + vector1.Get() * vector2.Get());
        ++outResult;
        ++vector1;
        ++vector2;
    }
    outResult -= vSize;
    vector1 -= vSize;
    vector2 -= vSize;
}

float Clip(float f,
           float absLimit)
{
    float result = (absLimit < f) ? absLimit : f;
    result = (-absLimit > result) ? -absLimit : result;
    return result;
}

void ClipVector(armnn::Decoder<float>& vector,
                uint32_t vSize,
                float absLimit,
                armnn::Encoder<float>& outResult)
{
    for (uint32_t v = 0; v < vSize; v++)
    {
        outResult.Set(Clip(vector.Get(), absLimit));
        ++vector;
        ++outResult;
    }
    vector -= vSize;
    outResult -= vSize;
}

void CopyVector(armnn::Decoder<float>& vector,
                uint32_t vSize,
                armnn::Encoder<float>& outResult)
{
    for (uint32_t v = 0; v < vSize; v++)
    {
        outResult.Set(vector.Get());
        ++outResult;
        ++vector;
    }
    outResult -= vSize;
    vector -= vSize;
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

std::unique_ptr<armnn::ScopedTensorHandle> AssignScopedTensorHandle(const armnn::ConstTensorHandle *ptr)
{
    if (!ptr)
    {
        return nullptr;
    }

    return std::make_unique<armnn::ScopedTensorHandle>(*ptr);
}
