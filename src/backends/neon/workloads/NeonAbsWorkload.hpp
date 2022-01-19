//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h>

namespace armnn
{

arm_compute::Status NeonAbsWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class NeonAbsWorkload : public NeonBaseWorkload<AbsQueueDescriptor>
{
public:
    NeonAbsWorkload(const AbsQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEAbsLayer m_AbsLayer;
};

} // namespace armnn
