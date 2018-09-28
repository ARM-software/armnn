//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefMeanUint8Workload.hpp"

#include "Mean.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

#include <vector>

namespace armnn
{

RefMeanUint8Workload::RefMeanUint8Workload(const MeanQueueDescriptor& descriptor, const WorkloadInfo& info)
  :Uint8Workload<MeanQueueDescriptor>(descriptor, info) {}


void RefMeanUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefMeanUint8Workload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    auto dequant = Dequantize(GetInputTensorDataU8(0, m_Data), inputInfo);

    std::vector<float> results(outputInfo.GetNumElements());

    Mean(inputInfo, outputInfo, m_Data.m_Parameters.m_Axis, dequant.data(), results.data());

    Quantize(GetOutputTensorDataU8(0, m_Data), results.data(), outputInfo);
}

} //namespace armnn

