//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonMergerFloatWorkload.hpp"

namespace armnn
{

void NeonMergerFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonMergerFloatWorkload_Execute");
    NeonBaseMergerWorkload::Execute();
}

} // namespace armnn
