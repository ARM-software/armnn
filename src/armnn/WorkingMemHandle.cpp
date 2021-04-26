//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "backendsCommon/CpuTensorHandle.hpp"
#include "WorkingMemHandle.hpp"
#include "Network.hpp"
#include <armnn/backends/IMemoryManager.hpp>

namespace armnn
{

namespace experimental
{

WorkingMemHandle::WorkingMemHandle(
        NetworkId networkId,
        std::vector<WorkingMemDescriptor> workingMemDescriptors,
        std::unordered_map<LayerGuid, WorkingMemDescriptor> workingMemDescriptorMap,
        std::vector<std::shared_ptr<IMemoryManager>> memoryManagers,
        std::unordered_map<LayerGuid, std::vector<std::unique_ptr<ITensorHandle> > > ownedTensorHandles) :
    m_NetworkId(networkId),
    m_WorkingMemDescriptors(workingMemDescriptors),
    m_WorkingMemDescriptorMap(workingMemDescriptorMap),
    m_MemoryManagers(memoryManagers),
    m_OwnedTensorHandles(std::move(ownedTensorHandles)),
    m_IsAllocated(false),
    m_Mutex()
{
}

void WorkingMemHandle::Allocate()
{
    if (m_IsAllocated)
    {
        return;
    }
    m_IsAllocated = true;

    for (auto& mgr : m_MemoryManagers)
    {
        mgr->Acquire();
    }
}

void WorkingMemHandle::Free()
{
    if (!m_IsAllocated)
    {
        return;
    }
    m_IsAllocated = false;

    for (auto& mgr : m_MemoryManagers)
    {
        mgr->Release();
    }
}

} // end experimental namespace

} // end armnn namespace
