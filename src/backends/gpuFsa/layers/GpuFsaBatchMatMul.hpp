//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Descriptors.hpp>

#include <gpuFsa/GpuFsaBackend.hpp>

namespace armnn
{
arm_compute::Status GpuFsaBatchMatMulValidate(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const BatchMatMulDescriptor& descriptor);

void GpuFsaBatchMatMulCreateOp(GpuFsaPreCompiledBlob* blob,
                               const TensorInfo& input0,
                               const TensorInfo& input1,
                               const BatchMatMulDescriptor& descriptor);

} // namespace armnn