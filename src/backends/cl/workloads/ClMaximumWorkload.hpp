//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLElementwiseOperations.h>

namespace armnn
{

arm_compute::Status ClMaximumWorkloadValidate(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const TensorInfo& output);

class ClMaximumWorkload : public ClBaseWorkload<MaximumQueueDescriptor>
{
public:
    ClMaximumWorkload(const MaximumQueueDescriptor& descriptor,
                      const WorkloadInfo& info,
                      const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;

private:
    mutable arm_compute::CLElementwiseMax m_MaximumLayer;
};

} //namespace armnn
