//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
