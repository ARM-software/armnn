//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/functions/CLElementwiseOperations.h>

namespace armnn
{

arm_compute::Status ClMaximumWorkloadValidate(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const TensorInfo& output);

class ClMaximumWorkload : public BaseWorkload<MaximumQueueDescriptor>
{
public:
    ClMaximumWorkload(const MaximumQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::CLElementwiseMax m_MaximumLayer;
};

} //namespace armnn
