//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h>

namespace armnn
{

arm_compute::Status NeonRsqrtWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class NeonRsqrtWorkload : public NeonBaseWorkload<RsqrtQueueDescriptor>
{
public:
    NeonRsqrtWorkload(const RsqrtQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NERsqrtLayer m_RsqrtLayer;
};

} // namespace armnn
