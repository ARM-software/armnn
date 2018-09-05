//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonMergerUint8Workload.hpp"

namespace armnn
{

void NeonMergerUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonMergerUint8Workload_Execute");
    NeonBaseMergerWorkload::Execute();
}

} // namespace armnn
