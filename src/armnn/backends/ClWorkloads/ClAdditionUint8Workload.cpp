//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
