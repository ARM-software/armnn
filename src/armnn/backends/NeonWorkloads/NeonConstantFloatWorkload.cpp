//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
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
