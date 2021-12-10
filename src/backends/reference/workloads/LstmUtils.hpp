//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "BaseIterator.hpp"
#include <armnn/backends/TensorHandle.hpp>

// Helper functions ported from the Android code base
// Refer to: android/external/tensorflow/tensorflow/contrib/lite/kernels/internal/reference/portable_tensor_utils.cc


void VectorBatchVectorAdd(armnn::Decoder<float>& vector,
                          uint32_t vSize,
                          armnn::Decoder<float>& batchVector,
                          uint32_t nBatch,
                          armnn::Encoder<float>& outResult );

// Layer norm for each batch.
// normalization_epsilon is added to avoid divergence.
void MeanStddevNormalization(armnn::Decoder<float>& input_vector,
                             armnn::Encoder<float>& output_vector,
                             uint32_t v_size,
                             uint32_t n_batch,
                             float normalization_epsilon);

void ZeroVector(armnn::Encoder<float>& vector,
                uint32_t vSize);

void MatrixBatchVectorMultiplyAccumulate(armnn::Decoder<float>& matrix,
                                         uint32_t mRows,
                                         uint32_t mCols,
                                         armnn::Decoder<float>& vector,
                                         uint32_t nBatch,
                                         armnn::Encoder<float>& outResult);

void VectorBatchVectorAssign(armnn::Decoder<float>& vector,
                             uint32_t vSize,
                             uint32_t nBatch,
                             armnn::Encoder<float>& outBatchVector);

void VectorBatchVectorCwiseProductAccumulate(armnn::Decoder<float>& vector,
                                             uint32_t vSize,
                                             armnn::Decoder<float>& batchVector,
                                             uint32_t nBatch,
                                             armnn::Encoder<float>& outResult);

void VectorBatchVectorCwiseProduct(armnn::Decoder<float>& vector,
                                   uint32_t vSize,
                                   armnn::Decoder<float>& batchVector,
                                   uint32_t nBatch,
                                   armnn::Encoder<float>& outResult);

void Sub1Vector(armnn::Decoder<float>& vector,
                uint32_t vSize,
                armnn::Encoder<float>& result);


void VectorVectorCwiseProduct(armnn::Decoder<float>& vector1,
                              armnn::Decoder<float>& vector2,
                              uint32_t vSize,
                              armnn::Encoder<float>& outResult);

void VectorVectorCwiseProductAccumulate(armnn::Decoder<float>& vector1,
                                        armnn::Decoder<float>& vector2,
                                        uint32_t vSize,
                                        armnn::Encoder<float>& outResult);

float Clip(float f,
           float absLimit);

void ClipVector(armnn::Decoder<float>& vector,
                uint32_t vSize,
                float absLimit,
                armnn::Encoder<float>& outResult);

void CopyVector(armnn::Decoder<float>& vector,
                uint32_t vSize,
                armnn::Encoder<float>& outResult);

void SetActivationParameters(uint32_t activation,
                             armnn::ActivationFunction& outArmnnActivation,
                             float& outA,
                             float& outB);

std::unique_ptr<armnn::ScopedTensorHandle> AssignScopedTensorHandle(const armnn::ConstTensorHandle *ptr);
