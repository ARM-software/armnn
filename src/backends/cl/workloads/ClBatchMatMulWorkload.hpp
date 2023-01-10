//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLMatMul.h>

namespace armnn
{
arm_compute::Status ClBatchMatMulValidate(const TensorInfo& inputX,
                                          const TensorInfo& inputY,
                                          const TensorInfo& output,
                                          const BatchMatMulDescriptor& descriptor,
                                          const ActivationDescriptor* activationDescriptor);

class ClBatchMatMulWorkload : public ClBaseWorkload<BatchMatMulQueueDescriptor>
{
public:
    ClBatchMatMulWorkload(const BatchMatMulQueueDescriptor& descriptor,
                          const WorkloadInfo& info,
                          const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLMatMul m_MatMulLayer;
};
} //namespace armnn
