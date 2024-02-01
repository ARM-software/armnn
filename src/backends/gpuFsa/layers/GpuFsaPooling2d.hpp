//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Descriptors.hpp>

#include <gpuFsa/GpuFsaBackend.hpp>

namespace armnn
{
arm_compute::Status GpuFsaPooling2dValidate(const TensorInfo& input,
                                            const Pooling2dDescriptor& descriptor);

void GpuFsaPooling2dCreateOp(GpuFsaPreCompiledBlob* blob,
                             const TensorInfo& input,
                             const Pooling2dDescriptor& descriptor);

} // namespace armnn