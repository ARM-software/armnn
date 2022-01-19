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

arm_compute::Status NeonLogicalNotWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class NeonLogicalNotWorkload : public NeonBaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    NeonLogicalNotWorkload(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NELogicalNot m_LogicalNotLayer;
};

} //namespace armnn
