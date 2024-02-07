//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>
#include <gpuFsa/GpuFsaBackend.hpp>

namespace armnn
{

arm_compute::Status GpuFsaReshapeValidate(const TensorInfo& input, const ReshapeDescriptor& descriptor);

void GpuFsaReshapeCreateOp(GpuFsaPreCompiledBlob* blob,
                           const TensorInfo& input,
                           const ReshapeDescriptor& descriptor);

} // namespace armnn

