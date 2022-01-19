//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/functions/CLElementwiseUnaryLayer.h>

namespace armnn
{

arm_compute::Status ClExpWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class ClExpWorkload : public ClBaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    ClExpWorkload(const ElementwiseUnaryQueueDescriptor& descriptor,
                  const WorkloadInfo& info,
                  const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLExpLayer m_ExpLayer;
};

} // namespace armnn
