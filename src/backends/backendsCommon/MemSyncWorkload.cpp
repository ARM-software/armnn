//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <ResolveType.hpp>

#include <backendsCommon/MemSyncWorkload.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>

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

} //namespace armnn
