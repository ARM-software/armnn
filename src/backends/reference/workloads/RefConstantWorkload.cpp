//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
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
    : RefBaseWorkload<ConstantQueueDescriptor>(descriptor, info) {}

void RefConstantWorkload::Execute() const
{
    Execute(m_Data.m_Outputs);
}

void RefConstantWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Outputs);
}

void RefConstantWorkload::Execute(std::vector<ITensorHandle*> outputs) const
{
    memcpy(outputs[0]->Map(), m_Data.m_LayerOutput->GetConstTensor<void>(), GetTensorInfo(outputs[0]).GetNumBytes());

    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefConstantWorkload_Execute");
}

} //namespace armnn
