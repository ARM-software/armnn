//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Tensor.hpp>
#include <armnn/Descriptors.hpp>

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLStridedSlice.h>

namespace armnn
{

arm_compute::Status ClStridedSliceWorkloadValidate(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const StridedSliceDescriptor& descriptor);

class ClStridedSliceWorkload : public ClBaseWorkload<StridedSliceQueueDescriptor>
{
public:
    ClStridedSliceWorkload(const StridedSliceQueueDescriptor& descriptor,
                           const WorkloadInfo& info,
                           const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;

private:
    mutable arm_compute::CLStridedSlice m_StridedSliceLayer;
};

} //namespace armnn
