//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefPooling2dUint8Workload.hpp"

#include "Pooling2d.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

#include <vector>

namespace armnn
{

void RefPooling2dUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefPooling2dUint8Workload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    auto dequant = Dequantize(GetInputTensorDataU8(0, m_Data), inputInfo);

    std::vector<float> results(outputInfo.GetNumElements());
    Pooling2d(dequant.data(),
              results.data(),
              inputInfo,
              outputInfo,
              m_Data.m_Parameters);

    Quantize(GetOutputTensorDataU8(0, m_Data), results.data(), outputInfo);
}

} //namespace armnn
