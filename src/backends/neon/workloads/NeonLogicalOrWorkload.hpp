//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/NEON/functions/NELogical.h>

namespace armnn
{

arm_compute::Status NeonLogicalOrWorkloadValidate(const TensorInfo& input0,
                                                  const TensorInfo& input1,
                                                  const TensorInfo& output);

class NeonLogicalOrWorkload : public NeonBaseWorkload<LogicalBinaryQueueDescriptor>
{
public:
    NeonLogicalOrWorkload(const LogicalBinaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NELogicalOr m_LogicalOrLayer;
};

} //namespace armnn
