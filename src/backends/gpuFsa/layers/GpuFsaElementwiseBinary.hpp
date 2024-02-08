//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Descriptors.hpp>

#include <gpuFsa/GpuFsaBackend.hpp>

namespace armnn
{
arm_compute::Status GpuFsaElementwiseBinaryValidate(const TensorInfo& input0,
                                                    const TensorInfo& input1,
                                                    const ElementwiseBinaryDescriptor& descriptor);

void GpuFsaElementwiseBinaryCreateOp(GpuFsaPreCompiledBlob* blob,
                                     const TensorInfo& input0,
                                     const TensorInfo& input1,
                                     const ElementwiseBinaryDescriptor& descriptor);

} // namespace armnn