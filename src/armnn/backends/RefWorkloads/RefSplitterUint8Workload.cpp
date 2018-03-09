//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
