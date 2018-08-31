//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
