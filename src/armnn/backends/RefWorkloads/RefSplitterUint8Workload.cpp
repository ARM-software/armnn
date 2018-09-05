//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefSplitterUint8Workload.hpp"

#include "Splitter.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefSplitterUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefSplitterUint8Workload_Execute");
    Splitter<uint8_t>(m_Data);
}

} //namespace armnn
