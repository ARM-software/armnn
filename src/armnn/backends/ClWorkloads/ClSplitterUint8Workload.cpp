//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClSplitterUint8Workload.hpp"

namespace armnn
{

void ClSplitterUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClSplitterUint8Workload_Execute");
    ClBaseSplitterWorkload::Execute();
}

} //namespace armnn
