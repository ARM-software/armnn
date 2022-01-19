//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/functions/CLArgMinMaxLayer.h>

namespace armnn
{

arm_compute::Status ClArgMinMaxWorkloadValidate(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const ArgMinMaxDescriptor& descriptor);

class ClArgMinMaxWorkload : public ClBaseWorkload<ArgMinMaxQueueDescriptor>
{
public:
    ClArgMinMaxWorkload(const ArgMinMaxQueueDescriptor& descriptor,
                        const WorkloadInfo& info,
                        const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLArgMinMaxLayer m_ArgMinMaxLayer;
};

} //namespace armnn
