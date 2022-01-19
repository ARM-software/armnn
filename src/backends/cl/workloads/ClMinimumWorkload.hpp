//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLElementwiseOperations.h>

namespace armnn
{

arm_compute::Status ClMinimumWorkloadValidate(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const TensorInfo& output);

class ClMinimumWorkload : public ClBaseWorkload<MinimumQueueDescriptor>
{
public:
    ClMinimumWorkload(const MinimumQueueDescriptor& descriptor,
                      const WorkloadInfo& info,
                      const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;

private:
    mutable arm_compute::CLElementwiseMin m_MinimumLayer;
};

} //namespace armnn
