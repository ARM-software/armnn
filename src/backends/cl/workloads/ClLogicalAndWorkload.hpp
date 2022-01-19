//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/functions/CLLogicalAnd.h>

namespace armnn
{

arm_compute::Status ClLogicalAndWorkloadValidate(const TensorInfo& input0,
                                                 const TensorInfo& input1,
                                                 const TensorInfo& output);

class ClLogicalAndWorkload : public ClBaseWorkload<LogicalBinaryQueueDescriptor>
{
public:
    ClLogicalAndWorkload(const LogicalBinaryQueueDescriptor& descriptor,
                         const WorkloadInfo& info,
                         const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLLogicalAnd m_LogicalAndLayer;
};

} //namespace armnn
