//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/runtime/NEON/functions/NEGather.h>

namespace armnn
{
arm_compute::Status NeonGatherWorkloadValidate(const TensorInfo& input,
                                               const TensorInfo& indices,
                                               const TensorInfo& output,
                                               const GatherDescriptor& descriptor);

class NeonGatherWorkload : public NeonBaseWorkload<GatherQueueDescriptor>
{
public:
    NeonGatherWorkload(const GatherQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NEGather m_Layer;
};

} //namespace armnn