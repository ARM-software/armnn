//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefSplitterFloat32Workload.hpp"

#include "Splitter.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefSplitterFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefSplitterFloat32Workload_Execute");
    Splitter<float>(m_Data);
}

} //namespace armnn
