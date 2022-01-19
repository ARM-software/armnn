//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLScale.h>

namespace armnn
{

arm_compute::Status ClResizeWorkloadValidate(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const ResizeDescriptor& descriptor);

class ClResizeWorkload : public ClBaseWorkload<ResizeQueueDescriptor>
{
public:
    ClResizeWorkload(const ResizeQueueDescriptor& descriptor,
                     const WorkloadInfo& info,
                     const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;

private:
    mutable arm_compute::CLScale m_ResizeLayer;
};

} // namespace armnn
