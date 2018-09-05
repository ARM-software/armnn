//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefResizeBilinearUint8Workload.hpp"

#include "RefWorkloadUtils.hpp"
#include "ResizeBilinear.hpp"

#include "Profiling.hpp"

#include <vector>

namespace armnn
{

void RefResizeBilinearUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefResizeBilinearUint8Workload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    auto dequant = Dequantize(GetInputTensorDataU8(0, m_Data), inputInfo);

    std::vector<float> results(outputInfo.GetNumElements());
    ResizeBilinear(dequant.data(), inputInfo, results.data(), outputInfo);

    Quantize(GetOutputTensorDataU8(0, m_Data), results.data(), outputInfo);
}

} //namespace armnn
