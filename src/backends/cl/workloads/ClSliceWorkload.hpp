//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/functions/CLSlice.h>

namespace armnn
{

arm_compute::Status ClSliceWorkloadValidate(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const SliceDescriptor& descriptor);

class ClSliceWorkload : public ClBaseWorkload<SliceQueueDescriptor>
{
public:
    ClSliceWorkload(const SliceQueueDescriptor& descriptor,
                    const WorkloadInfo& info,
                    const arm_compute::CLCompileContext& clCompileContext);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLSlice m_SliceFunction;
};

} // namespace armnn
