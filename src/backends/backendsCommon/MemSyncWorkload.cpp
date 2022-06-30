//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <ResolveType.hpp>

#include <backendsCommon/MemSyncWorkload.hpp>
#include <armnn/backends/TensorHandle.hpp>

#include <cstring>

namespace armnn
{

SyncMemGenericWorkload::SyncMemGenericWorkload(const MemSyncQueueDescriptor& descriptor,
                                               const WorkloadInfo& info)
    : BaseWorkload<MemSyncQueueDescriptor>(descriptor, info)
{
    m_TensorHandle = descriptor.m_Inputs[0];
}

void SyncMemGenericWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "SyncMemGeneric_Execute");
    m_TensorHandle->Map(true);
    m_TensorHandle->Unmap();
}

void SyncMemGenericWorkload::ExecuteAsync(ExecutionData& executionData)
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "SyncMemGeneric_Execute_WorkingMemDescriptor");

    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    workingMemDescriptor->m_Inputs[0]->Map(true);
    workingMemDescriptor->m_Inputs[0]->Unmap();
}

} //namespace armnn
