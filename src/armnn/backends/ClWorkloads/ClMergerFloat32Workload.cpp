//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClMergerFloat32Workload.hpp"


namespace armnn
{

void ClMergerFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClMergerFloat32Workload_Execute");
    ClBaseMergerWorkload::Execute();
}

} //namespace armnn

