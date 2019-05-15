//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefMergerWorkload.hpp"

#include "Merger.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefMergerWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefMergerWorkload_Execute");
    Merger(m_Data);
}

} //namespace armnn
