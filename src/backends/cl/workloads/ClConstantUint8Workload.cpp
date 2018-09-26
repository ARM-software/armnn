//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClConstantUint8Workload.hpp"
#include "ClWorkloadUtils.hpp"

namespace armnn
{

void ClConstantUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClConstantUint8Workload_Execute");
    ClBaseConstantWorkload::Execute();
}

} //namespace armnn
