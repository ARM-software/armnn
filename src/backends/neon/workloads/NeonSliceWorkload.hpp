//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NeonBaseWorkload.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/NEON/functions/NESlice.h>

namespace armnn
{

arm_compute::Status NeonSliceWorkloadValidate(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const SliceDescriptor& descriptor);

class NeonSliceWorkload : public NeonBaseWorkload<SliceQueueDescriptor>
{
public:
    NeonSliceWorkload(const SliceQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NESlice m_SliceFunction;
};

} // namespace armnn