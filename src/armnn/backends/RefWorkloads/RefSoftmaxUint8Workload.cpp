//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefSoftmaxUint8Workload.hpp"

#include "RefWorkloadUtils.hpp"
#include "Softmax.hpp"

#include "Profiling.hpp"

#include <vector>

namespace armnn
{

void RefSoftmaxUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefSoftmaxUint8Workload_Execute");

    const TensorInfo& tensorInfo = GetTensorInfo(m_Data.m_Inputs[0]);

    auto dequant = Dequantize(GetInputTensorDataU8(0, m_Data), tensorInfo);

    std::vector<float> results(tensorInfo.GetNumElements());

    Softmax(dequant.data(),
            results.data(),
            tensorInfo,
            m_Data.m_Parameters.m_Beta);

    Quantize(GetOutputTensorDataU8(0, m_Data), results.data(), GetTensorInfo(m_Data.m_Outputs[0]));
}

} //namespace armnn
