//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/functions/CLElementwiseUnaryLayer.h>

namespace armnn
{

arm_compute::Status ClLogWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class ClLogWorkload : public ClBaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    ClLogWorkload(const ElementwiseUnaryQueueDescriptor& descriptor,
                  const WorkloadInfo& info,
                  const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLLogLayer m_LogLayer;
};

} // namespace armnn
