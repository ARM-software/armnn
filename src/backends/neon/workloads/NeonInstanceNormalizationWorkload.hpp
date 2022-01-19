//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <neon/workloads/NeonWorkloadUtils.hpp>

#include <arm_compute/runtime/NEON/functions/NEInstanceNormalizationLayer.h>

namespace armnn
{

arm_compute::Status NeonInstanceNormalizationWorkloadValidate(const TensorInfo& input,
                                                              const TensorInfo& output,
                                                              const InstanceNormalizationDescriptor& descriptor);

class NeonInstanceNormalizationWorkload : public NeonBaseWorkload<InstanceNormalizationQueueDescriptor>
{
public:
    NeonInstanceNormalizationWorkload(const InstanceNormalizationQueueDescriptor& descriptor,
                                      const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::NEInstanceNormalizationLayer m_Layer;
};

} // namespace armnn