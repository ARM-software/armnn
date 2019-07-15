//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/NEON/functions/NEScale.h>

namespace armnn
{

arm_compute::Status NeonResizeWorkloadValidate(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const ResizeDescriptor& descriptor);

class NeonResizeWorkload : public BaseWorkload<ResizeQueueDescriptor>
{
public:
    NeonResizeWorkload(const ResizeQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::NEScale m_ResizeLayer;
};

} //namespace armnn
