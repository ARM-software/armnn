//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/WorkloadData.hpp>
#include "ClBaseWorkload.hpp"
#include <arm_compute/runtime/CL/functions/CLFill.h>

namespace armnn {

class ClFillWorkload : public ClBaseWorkload<FillQueueDescriptor>
{
public:
    ClFillWorkload(const FillQueueDescriptor& descriptor,
                   const WorkloadInfo& info,
                   const arm_compute::CLCompileContext& clCompileContext);
    void Execute() const override;

private:
    mutable arm_compute::CLFill m_Layer;
};

} //namespace armnn
