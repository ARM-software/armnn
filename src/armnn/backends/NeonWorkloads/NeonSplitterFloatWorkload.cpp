//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonSplitterFloatWorkload.hpp"

namespace armnn
{

void NeonSplitterFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonSplitterFloatWorkload_Execute");
    NeonBaseSplitterWorkload::Execute();
}

} //namespace armnn
