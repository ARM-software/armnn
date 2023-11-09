//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ClBaseWorkload.hpp"

#include <arm_compute/runtime/CL/functions/CLReverse.h>
#include <arm_compute/runtime/Tensor.h>
#include "arm_compute/runtime/CL/CLTensor.h"

namespace armnn
{
arm_compute::Status ClReverseV2WorkloadValidate(const TensorInfo& input,
                                                const TensorInfo& axis,
                                                const TensorInfo& output);

class ClReverseV2Workload : public BaseWorkload<ReverseV2QueueDescriptor> 
{
public:
    ClReverseV2Workload(const ReverseV2QueueDescriptor &descriptor,
                        const WorkloadInfo &info,
                        const arm_compute::CLCompileContext& clCompileContext);

    void Execute() const override;

private:
    mutable arm_compute::CLReverse m_Layer;
};

} //namespace armnn