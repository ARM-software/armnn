//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLPReluLayer.h>

namespace armnn
{
arm_compute::Status ClPreluWorkloadValidate(const TensorInfo& input,
                                            const TensorInfo& alpha,
                                            const TensorInfo& output);

class ClPreluWorkload : public ClBaseWorkload<PreluQueueDescriptor>
{
public:
    ClPreluWorkload(const PreluQueueDescriptor& descriptor,
                    const WorkloadInfo& info,
                    const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;

private:
    mutable arm_compute::CLPReluLayer m_PreluLayer;
};

} //namespace armnn
