//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClSplitterFloat32Workload.hpp"

namespace armnn
{

void ClSplitterFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClSplitterFloat32Workload_Execute");
    ClBaseSplitterWorkload::Execute();
}

} //namespace armnn
