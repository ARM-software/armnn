//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
