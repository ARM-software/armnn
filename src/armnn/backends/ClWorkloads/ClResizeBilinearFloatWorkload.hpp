//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "backends/Workload.hpp"

#include <arm_compute/runtime/CL/CLFunctions.h>

namespace armnn
{

class ClResizeBilinearFloatWorkload : public FloatWorkload<ResizeBilinearQueueDescriptor>
{
public:
    ClResizeBilinearFloatWorkload(const ResizeBilinearQueueDescriptor& descriptor, const WorkloadInfo& info);
    void Execute() const override;

private:
    mutable arm_compute::CLScale m_ResizeBilinearLayer;
};

} //namespace armnn
