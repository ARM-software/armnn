//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonMergerFloat32Workload.hpp"

namespace armnn
{

void NeonMergerFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "ClMergerFloat32Workload_Execute");
    NeonBaseMergerWorkload::Execute();
}

} // namespace armnn
