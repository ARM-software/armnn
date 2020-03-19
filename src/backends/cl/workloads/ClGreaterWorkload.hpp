//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLComparison.h>

namespace armnn
{

arm_compute::Status ClGreaterWorkloadValidate(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const TensorInfo& output);

class ClGreaterWorkload : public BaseWorkload<GreaterQueueDescriptor>
{
public:
    ClGreaterWorkload(const GreaterQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::CLComparison m_GreaterLayer;
};

} //namespace armnn
