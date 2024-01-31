//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Descriptors.hpp>

#include <gpuFsa/GpuFsaBackend.hpp>

namespace armnn
{

    using namespace arm_compute::experimental::dynamic_fusion;

    arm_compute::Status GpuFsaElementwiseBinarySubValidate(const TensorInfo& input0,
                                                           const TensorInfo& input1);

    void GpuFsaElementwiseBinarySubCreateOp(GpuFsaPreCompiledBlob* blob,
                                            const TensorInfo& input0,
                                            const TensorInfo& input1);
}