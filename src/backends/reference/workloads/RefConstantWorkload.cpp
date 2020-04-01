//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefConstantWorkload.hpp"

#include "RefWorkloadUtils.hpp"

#include <armnn/Types.hpp>

#include <armnn/utility/Assert.hpp>

#include <cstring>

namespace armnn
{

RefConstantWorkload::RefConstantWorkload(
    const ConstantQueueDescriptor& descriptor, const WorkloadInfo& info)
    : BaseWorkload<ConstantQueueDescriptor>(descriptor, info) {}

void RefConstantWorkload::PostAllocationConfigure()
{
    const ConstantQueueDescriptor& data = this->m_Data;

    ARMNN_ASSERT(data.m_LayerOutput != nullptr);

    const TensorInfo& outputInfo = GetTensorInfo(data.m_Outputs[0]);
    ARMNN_ASSERT(data.m_LayerOutput->GetTensorInfo().GetNumBytes() == outputInfo.GetNumBytes());

    memcpy(GetOutputTensorData<void>(0, data), data.m_LayerOutput->GetConstTensor<void>(),
        outputInfo.GetNumBytes());
}

void RefConstantWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefConstantWorkload_Execute");
}

} //namespace armnn
