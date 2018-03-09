//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "RefResizeBilinearFloat32Workload.hpp"

#include "RefWorkloadUtils.hpp"
#include "ResizeBilinear.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefResizeBilinearFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefResizeBilinearFloat32Workload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    ResizeBilinear(GetInputTensorDataFloat(0, m_Data),
        inputInfo,
        GetOutputTensorDataFloat(0, m_Data),
        outputInfo);
}

} //namespace armnn
