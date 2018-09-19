//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefConstantUint8Workload.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefConstantUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefConstantUint8Workload_Execute");
    RefBaseConstantWorkload::Execute();
}

} //namespace armnn
