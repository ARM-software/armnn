//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
