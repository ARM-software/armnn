//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <neon/workloads/NeonWorkloadUtils.hpp>

#include <arm_compute/runtime/NEON/functions/NEElementwiseOperations.h>

namespace armnn
{

arm_compute::Status NeonMaximumWorkloadValidate(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output);

class NeonMaximumWorkload : public NeonBaseWorkload<MaximumQueueDescriptor>
{
public:
    NeonMaximumWorkload(const MaximumQueueDescriptor& descriptor, const WorkloadInfo& info);

    virtual void Execute() const override;

private:
    mutable arm_compute::NEElementwiseMax m_MaxLayer;
};

} //namespace armnn
