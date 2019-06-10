//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefMeanWorkload.hpp"

#include "Mean.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

#include <vector>

namespace armnn
{

RefMeanWorkload::RefMeanWorkload(const MeanQueueDescriptor& descriptor, const WorkloadInfo& info)
  :BaseWorkload<MeanQueueDescriptor>(descriptor, info) {}

void RefMeanWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefMeanWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    auto inputDecoder  = MakeDecoder<float>(inputInfo,  m_Data.m_Inputs[0]->Map());
    auto outputEncoder = MakeEncoder<float>(outputInfo, m_Data.m_Outputs[0]->Map());

    Mean(inputInfo, outputInfo, m_Data.m_Parameters.m_Axis, *inputDecoder, *outputEncoder);
}

} //namespace armnn
