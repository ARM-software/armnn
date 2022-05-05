//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"
#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/NEON/functions/NEActivationLayer.h>

namespace armnn
{

arm_compute::Status NeonSqrtWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class NeonSqrtWorkload : public NeonBaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    NeonSqrtWorkload(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEActivationLayer m_SqrtLayer;
};

} //namespace armnn