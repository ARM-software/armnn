//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Descriptors.hpp>

#include <gpuFsa/GpuFsaBackend.hpp>

namespace armnn
{
arm_compute::Status GpuFsaActivationValidate(const TensorInfo& input,
                                             const ActivationDescriptor& descriptor);

void GpuFsaActivationCreateOp(GpuFsaPreCompiledBlob* blob,
                              const TensorInfo& input,
                              const ActivationDescriptor& descriptor);

} // namespace armnn