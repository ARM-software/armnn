//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClMergerUint8Workload.hpp"


namespace armnn
{

void ClMergerUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "ClMergerUint8Workload_Execute");
    ClBaseMergerWorkload<DataType::QuantisedAsymm8>::Execute();
}

} //namespace armnn
