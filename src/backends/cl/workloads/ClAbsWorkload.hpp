//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/functions/CLElementWiseUnaryLayer.h>

namespace armnn
{

arm_compute::Status ClAbsWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class ClAbsWorkload : public BaseWorkload<AbsQueueDescriptor>
{
public:
    ClAbsWorkload(const AbsQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLAbsLayer m_AbsLayer;
};

} // namespace armnn
