//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonConstantFloat32Workload.hpp"

namespace armnn
{

void NeonConstantFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "NeonConstantFloat32Workload_Execute");
    NeonBaseConstantWorkload::Execute();
}

} //namespace armnn
