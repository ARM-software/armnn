//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
