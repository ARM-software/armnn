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

arm_compute::Status ClNegWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class ClNegWorkload : public ClBaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    ClNegWorkload(const ElementwiseUnaryQueueDescriptor& descriptor,
                  const WorkloadInfo& info,
                  const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLNegLayer m_NegLayer;
};

} // namespace armnn
