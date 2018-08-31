//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonConstantFloatWorkload.hpp"

namespace armnn
{

void NeonConstantFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonConstantFloatWorkload_Execute");
    NeonBaseConstantWorkload::Execute();
}

} //namespace armnn
