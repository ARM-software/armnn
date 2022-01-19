//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/functions/CLElementwiseUnaryLayer.h>

namespace armnn
{

arm_compute::Status ClRsqrtWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class ClRsqrtWorkload : public ClBaseWorkload<RsqrtQueueDescriptor>
{
public:
    ClRsqrtWorkload(const RsqrtQueueDescriptor& descriptor,
                    const WorkloadInfo& info,
                    const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLRsqrtLayer m_RsqrtLayer;
};

} // namespace armnn
