//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>

#include <arm_compute/core/Error.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h>

namespace armnn
{

using namespace arm_compute::experimental::dynamic_fusion;

arm_compute::Status GpuFsaConvolution2dValidate(const TensorInfo& input,
                                                const Convolution2dDescriptor& descriptor,
                                                const TensorInfo& weights,
                                                const Optional<TensorInfo>& biases);

void GpuFsaConvolution2dCreateOp(const TensorInfo& input,
                                 const Convolution2dDescriptor& descriptor,
                                 const TensorInfo& weights,
                                 const Optional<TensorInfo>& biases);

} // namespace armnn