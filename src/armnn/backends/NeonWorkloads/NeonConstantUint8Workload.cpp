//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonConstantUint8Workload.hpp"

namespace armnn
{

void NeonConstantUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonConstantUint8Workload_Execute");
    NeonBaseConstantWorkload::Execute();
}

} //namespace armnn
