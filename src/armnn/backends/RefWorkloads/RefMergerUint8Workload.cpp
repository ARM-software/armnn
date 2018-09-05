//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefMergerUint8Workload.hpp"

#include "Merger.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefMergerUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefMergerUint8Workload_Execute");
    Merger<uint8_t>(m_Data);
}

} //namespace armnn
