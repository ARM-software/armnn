//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/functions/CLActivationLayer.h>

namespace armnn
{

arm_compute::Status ClSqrtWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class ClSqrtWorkload : public ClBaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    ClSqrtWorkload(const ElementwiseUnaryQueueDescriptor& descriptor,
                   const WorkloadInfo& info,
                   const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLActivationLayer m_SqrtLayer;
};

} // namespace armnn
