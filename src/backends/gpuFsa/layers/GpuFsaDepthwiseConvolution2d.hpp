//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Descriptors.hpp>

#include <gpuFsa/GpuFsaBackend.hpp>

namespace armnn
{
arm_compute::Status GpuFsaDepthwiseConvolution2dValidate(const TensorInfo& input,
                                                         const DepthwiseConvolution2dDescriptor& descriptor,
                                                         const TensorInfo& weights,
                                                         const Optional<TensorInfo>& biases);

void GpuFsaDepthwiseConvolution2dCreateOp(GpuFsaPreCompiledBlob* blob,
                                          const TensorInfo& input,
                                          const DepthwiseConvolution2dDescriptor& descriptor,
                                          const TensorInfo& weights,
                                          const Optional<TensorInfo>& biases);

} // namespace armnn
