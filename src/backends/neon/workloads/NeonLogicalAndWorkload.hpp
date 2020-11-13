//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/NEON/functions/NELogical.h>

namespace armnn
{

arm_compute::Status NeonLogicalAndWorkloadValidate(const TensorInfo& input0,
                                                   const TensorInfo& input1,
                                                   const TensorInfo& output);

class NeonLogicalAndWorkload : public BaseWorkload<LogicalBinaryQueueDescriptor>
{
public:
    NeonLogicalAndWorkload(const LogicalBinaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NELogicalAnd m_LogicalAndLayer;
};

} //namespace armnn
