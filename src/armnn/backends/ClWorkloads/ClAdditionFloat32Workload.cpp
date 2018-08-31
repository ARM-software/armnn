//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClAdditionFloat32Workload.hpp"

#include "backends/ClTensorHandle.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/ArmComputeTensorUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

void ClAdditionFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClAdditionFloat32Workload_Execute");
    ClAdditionBaseWorkload::Execute();
}

} //namespace armnn
