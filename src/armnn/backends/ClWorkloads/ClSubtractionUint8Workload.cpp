//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClSubtractionUint8Workload.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

void ClSubtractionUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClSubtractionUint8Workload_Execute");
    ClSubtractionBaseWorkload::Execute();
}

} //namespace armnn
