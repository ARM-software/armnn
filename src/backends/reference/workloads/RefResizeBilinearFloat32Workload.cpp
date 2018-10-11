//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
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
        outputInfo,
        m_Data.m_Parameters.m_DataLayout);
}

} //namespace armnn
