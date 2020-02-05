//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLScale.h>

namespace armnn
{

arm_compute::Status ClResizeWorkloadValidate(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const ResizeDescriptor& descriptor);

class ClResizeWorkload : public BaseWorkload<ResizeQueueDescriptor>
{
public:
    ClResizeWorkload(const ResizeQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::CLScale m_ResizeLayer;
};

} // namespace armnn
