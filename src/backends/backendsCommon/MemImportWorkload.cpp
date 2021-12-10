//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <ResolveType.hpp>

#include <backendsCommon/MemImportWorkload.hpp>
#include <armnn/backends/TensorHandle.hpp>

#include <cstring>

namespace armnn
{

ImportMemGenericWorkload::ImportMemGenericWorkload(const MemImportQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info)
    : BaseWorkload<MemImportQueueDescriptor>(descriptor, info)
{
    m_TensorHandlePairs = std::make_pair(descriptor.m_Inputs[0], descriptor.m_Outputs[0]);
}

void ImportMemGenericWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ImportMemGeneric_Execute");

    m_TensorHandlePairs.second->Import(const_cast<void*>(m_TensorHandlePairs.first->Map(true)), MemorySource::Malloc);
    m_TensorHandlePairs.first->Unmap();
}

} //namespace armnn
