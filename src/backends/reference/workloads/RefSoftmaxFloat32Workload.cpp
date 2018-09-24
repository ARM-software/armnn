//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefSoftmaxFloat32Workload.hpp"

#include "RefWorkloadUtils.hpp"
#include "Softmax.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefSoftmaxFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefSoftmaxFloat32Workload_Execute");

    Softmax(GetInputTensorDataFloat(0, m_Data),
            GetOutputTensorDataFloat(0, m_Data),
            GetTensorInfo(m_Data.m_Inputs[0]),
            m_Data.m_Parameters.m_Beta);
}

} //namespace armnn
