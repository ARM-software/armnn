//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClConstantUint8Workload.hpp"
namespace armnn
{

void ClConstantUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClConstantUint8Workload_Execute");
    ClBaseConstantWorkload::Execute();
}

} //namespace armnn