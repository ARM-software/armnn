//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefRsqrtFloat32Workload.hpp"

#include "RefWorkloadUtils.hpp"
#include "Rsqrt.hpp"

#include <Profiling.hpp>

namespace armnn
{

void RefRsqrtFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefRsqrtFloat32Workload_Execute");

    Rsqrt(GetInputTensorDataFloat(0, m_Data),
          GetOutputTensorDataFloat(0, m_Data),
          GetTensorInfo(m_Data.m_Inputs[0]));
}

} //namespace armnn
