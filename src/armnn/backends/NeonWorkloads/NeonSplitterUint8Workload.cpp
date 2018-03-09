//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonSplitterUint8Workload.hpp"

namespace armnn
{

void NeonSplitterUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "NeonSplitterUint8Workload_Execute");
    NeonBaseSplitterWorkload::Execute();
}

} //namespace armnn
