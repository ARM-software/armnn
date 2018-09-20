//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backends/Workload.hpp"

#include <arm_compute/runtime/CL/CLFunctions.h>

namespace armnn
{

class ClReshapeFloatWorkload : public FloatWorkload<ReshapeQueueDescriptor>
{
public:
    ClReshapeFloatWorkload(const ReshapeQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLReshapeLayer m_Layer;
};

} //namespace armnn


