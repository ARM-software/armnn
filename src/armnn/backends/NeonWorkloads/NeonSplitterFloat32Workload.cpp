//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonSplitterFloat32Workload.hpp"

namespace armnn
{

void NeonSplitterFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonSplitterFloat32Workload_Execute");
    NeonBaseSplitterWorkload::Execute();
}

} //namespace armnn
