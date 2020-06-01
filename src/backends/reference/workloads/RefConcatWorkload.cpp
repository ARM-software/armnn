//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefConcatWorkload.hpp"

#include "Concatenate.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefConcatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefConcatWorkload_Execute");
    Concatenate(m_Data);
}

} //namespace armnn
