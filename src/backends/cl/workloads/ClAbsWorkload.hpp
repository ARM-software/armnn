//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/functions/CLElementwiseUnaryLayer.h>

namespace armnn
{

arm_compute::Status ClAbsWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class ClAbsWorkload : public ClBaseWorkload<AbsQueueDescriptor>
{
public:
    ClAbsWorkload(const AbsQueueDescriptor& descriptor,
                  const WorkloadInfo& info,
                  const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLAbsLayer m_AbsLayer;
};

} // namespace armnn
