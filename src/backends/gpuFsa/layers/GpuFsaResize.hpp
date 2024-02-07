//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Descriptors.hpp>

#include <gpuFsa/GpuFsaBackend.hpp>

namespace armnn
{
arm_compute::Status GpuFsaResizeValidate(const TensorInfo& input,
                                         const ResizeDescriptor& descriptor);

void GpuFsaResizeCreateOp(GpuFsaPreCompiledBlob* blob,
                          const TensorInfo& input,
                          const ResizeDescriptor& descriptor);

} // namespace armnn