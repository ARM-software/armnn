//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClConstantFloatWorkload.hpp"
namespace armnn
{

void ClConstantFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClConstantFloatWorkload_Execute");
    ClBaseConstantWorkload::Execute();
}

} //namespace armnn