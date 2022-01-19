//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLReductionOperation.h>

namespace armnn
{

arm_compute::Status ClReduceWorkloadValidate(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const ReduceDescriptor& descriptor);

class ClReduceWorkload : public ClBaseWorkload<ReduceQueueDescriptor>
{
public:
    ClReduceWorkload(const ReduceQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLReductionOperation m_Layer;
};

} //namespace armnn
