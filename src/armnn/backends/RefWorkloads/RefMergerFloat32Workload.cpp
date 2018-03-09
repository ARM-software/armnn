//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "RefMergerFloat32Workload.hpp"

#include "Merger.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefMergerFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefMergerFloat32Workload_Execute");
    Merger<float>(m_Data);
}

} //namespace armnn
