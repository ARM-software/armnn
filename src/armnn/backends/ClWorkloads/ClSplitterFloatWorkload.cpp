//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
