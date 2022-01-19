//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/runtime/NEON/functions/NEReductionOperation.h>

namespace armnn
{

arm_compute::Status NeonReduceWorkloadValidate(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const ReduceDescriptor& descriptor);

class NeonReduceWorkload : public NeonBaseWorkload<ReduceQueueDescriptor>
{
public:
    NeonReduceWorkload(const ReduceQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::NEReductionOperation m_Layer;
};

} //namespace armnn
