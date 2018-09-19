//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefActivationFloat32Workload.hpp"

#include "Activation.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefActivationFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefActivationFloat32Workload_Execute");

    Activation(GetInputTensorDataFloat(0, m_Data),
               GetOutputTensorDataFloat(0, m_Data),
               GetTensorInfo(m_Data.m_Inputs[0]),
               m_Data.m_Parameters.m_Function,
               m_Data.m_Parameters.m_A,
               m_Data.m_Parameters.m_B);
}

} //namespace armnn
