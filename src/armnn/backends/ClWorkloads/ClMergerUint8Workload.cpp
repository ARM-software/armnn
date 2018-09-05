//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClMergerUint8Workload.hpp"


namespace armnn
{

void ClMergerUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClMergerUint8Workload_Execute");
    ClBaseMergerWorkload<DataType::QuantisedAsymm8>::Execute();
}

} //namespace armnn
