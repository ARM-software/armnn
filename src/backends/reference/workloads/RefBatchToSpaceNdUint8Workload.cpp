//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchToSpaceNd.hpp"
#include "Profiling.hpp"
#include "RefBatchToSpaceNdUint8Workload.hpp"
#include "RefWorkloadUtils.hpp"

namespace armnn
{

void RefBatchToSpaceNdUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefBatchToSpaceNdUint8Workload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);
    auto dequantizedInputData = Dequantize(GetInputTensorDataU8(0, m_Data), inputInfo);

    std::vector<float> results(outputInfo.GetNumElements());
    BatchToSpaceNd(m_Data.m_Parameters.m_DataLayout, inputInfo, outputInfo, m_Data.m_Parameters.m_BlockShape,
                   m_Data.m_Parameters.m_Crops, dequantizedInputData.data(), results.data());

    Quantize(GetOutputTensorDataU8(0, m_Data), results.data(), outputInfo);
}

} //namespace armnn