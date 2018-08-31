//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClMergerFloatWorkload.hpp"


namespace armnn
{

void ClMergerFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClMergerFloatWorkload_Execute");
    ClBaseMergerWorkload::Execute();
}

} //namespace armnn

