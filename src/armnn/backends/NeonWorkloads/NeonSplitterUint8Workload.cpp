//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonSplitterUint8Workload.hpp"

namespace armnn
{

void NeonSplitterUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonSplitterUint8Workload_Execute");
    NeonBaseSplitterWorkload::Execute();
}

} //namespace armnn
