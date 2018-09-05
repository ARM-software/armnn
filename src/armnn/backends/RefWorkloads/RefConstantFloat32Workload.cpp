//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefConstantFloat32Workload.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefConstantFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefConstantFloat32Workload_Execute");
    RefBaseConstantWorkload::Execute();
}

} //namespace armnn
