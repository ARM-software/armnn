//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h>

namespace armnn
{

arm_compute::Status NeonAbsWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class NeonAbsWorkload : public BaseWorkload<AbsQueueDescriptor>
{
public:
    NeonAbsWorkload(const AbsQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEAbsLayer m_AbsLayer;
};

} // namespace armnn
