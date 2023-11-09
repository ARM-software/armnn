//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/runtime/NEON/functions/NEReverse.h>
#include <arm_compute/runtime/Tensor.h>

namespace armnn
{
arm_compute::Status NeonReverseV2WorkloadValidate(const TensorInfo& input,
                                                  const TensorInfo& axis,
                                                  const TensorInfo& output);

class NeonReverseV2Workload : public BaseWorkload<ReverseV2QueueDescriptor>
{
public:
    NeonReverseV2Workload(const ReverseV2QueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::NEReverse m_Layer;
};

} // namespace armnn