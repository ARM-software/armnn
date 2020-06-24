//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/Workload.hpp>
#include <arm_compute/runtime/CL/functions/CLFill.h>

namespace armnn {

class ClFillWorkload : public BaseWorkload<FillQueueDescriptor>
{
public:
    ClFillWorkload(const FillQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::CLFill m_Layer;
};

} //namespace armnn
