//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefActivationUint8Workload.hpp"

#include "Activation.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

#include <vector>

namespace armnn
{

void RefActivationUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefActivationUint8Workload_Execute");

    const TensorInfo& tensorInfo = GetTensorInfo(m_Data.m_Inputs[0]);

    auto dequant = Dequantize(GetInputTensorDataU8(0, m_Data), tensorInfo);

    std::vector<float> results(tensorInfo.GetNumElements());

    Activation(dequant.data(),
               results.data(),
               tensorInfo,
               m_Data.m_Parameters.m_Function,
               m_Data.m_Parameters.m_A,
               m_Data.m_Parameters.m_B);

    Quantize(GetOutputTensorDataU8(0, m_Data), results.data(), GetTensorInfo(m_Data.m_Outputs[0]));
}

} //namespace armnn
