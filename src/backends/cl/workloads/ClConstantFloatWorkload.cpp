//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClConstantFloatWorkload.hpp"
#include "ClWorkloadUtils.hpp"

namespace armnn
{

void ClConstantFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClConstantFloatWorkload_Execute");
    ClBaseConstantWorkload::Execute();
}

} //namespace armnn
