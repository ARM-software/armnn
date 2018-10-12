//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefPadWorkload.hpp"

#include "Pad.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

#include <vector>

namespace armnn
{

RefPadWorkload::RefPadWorkload(const PadQueueDescriptor& descriptor, const WorkloadInfo& info)
  :BaseWorkload<PadQueueDescriptor>(descriptor, info) {}


void RefPadWorkload::Execute() const
{

    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefPadWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    const float* inputData = GetInputTensorDataFloat(0, m_Data);
    float* outputData = GetOutputTensorDataFloat(0, m_Data);


    Pad(inputInfo, outputInfo, m_Data.m_Parameters.m_PadList, inputData, outputData);
}

} //namespace armnn