//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClMergerFloatWorkload.hpp"

#include "ClWorkloadUtils.hpp"

namespace armnn
{

void ClMergerFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClMergerFloatWorkload_Execute");
    ClBaseMergerWorkload::Execute();
}

} //namespace armnn

