//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClAdditionUint8Workload.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

void ClAdditionUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClAdditionUint8Workload_Execute");
    ClAdditionBaseWorkload::Execute();
}

} //namespace armnn
