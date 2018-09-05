//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClSplitterFloatWorkload.hpp"

namespace armnn
{

void ClSplitterFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClSplitterFloatWorkload_Execute");
    ClBaseSplitterWorkload::Execute();
}

} //namespace armnn
