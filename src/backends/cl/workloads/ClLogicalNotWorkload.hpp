//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/functions/CLLogicalNot.h>

namespace armnn
{

arm_compute::Status ClLogicalNotWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class ClLogicalNotWorkload : public ClBaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    ClLogicalNotWorkload(const ElementwiseUnaryQueueDescriptor& descriptor,
                         const WorkloadInfo& info,
                         const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLLogicalNot m_LogicalNotLayer;
};

} //namespace armnn
