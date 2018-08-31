//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
