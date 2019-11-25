//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/CL/functions/CLSlice.h>

namespace armnn
{

arm_compute::Status ClSliceWorkloadValidate(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const SliceDescriptor& descriptor);

class ClSliceWorkload : public BaseWorkload<SliceQueueDescriptor>
{
public:
    ClSliceWorkload(const SliceQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::CLSlice m_SliceFunction;
};

} // namespace armnn
