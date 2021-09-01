//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "backendsCommon/TensorHandle.hpp"
#include "WorkingMemHandle.hpp"
#include "Network.hpp"
#include <armnn/backends/IMemoryManager.hpp>

namespace armnn
{

namespace experimental
{

WorkingMemHandle::WorkingMemHandle(NetworkId networkId,
        std::vector<std::pair<LayerBindingId, LayerGuid>> inputHandles,
        std::vector<InputConnectionInfo> inputConnections,
        std::vector<WorkingMemDescriptor> workingMemDescriptors,
        std::unordered_map<LayerGuid, WorkingMemDescriptor> workingMemDescriptorMap,
        std::vector<std::shared_ptr<IMemoryManager>> memoryManagers,
        std::unordered_map<LayerGuid, std::vector<std::unique_ptr<ITensorHandle> > > ownedTensorHandles)
    : m_NetworkId(networkId)
    , m_WorkingMemDescriptors(workingMemDescriptors)
    , m_WorkingMemDescriptorMap(workingMemDescriptorMap)
    , m_MemoryManagers(memoryManagers)
    , m_OwnedTensorHandles(std::move(ownedTensorHandles))
    , m_IsAllocated(false)
    , m_Mutex()
{
    unsigned int maxInputBindingId = 0;
    for (auto pair : inputHandles)
    {
        unsigned int bindingId = numeric_cast<unsigned int>(pair.first);
        if (maxInputBindingId < bindingId)
        {
            maxInputBindingId = bindingId;
        }

    }

    // Create a map of LayerBindingIds to the corresponding input ITensorHandle*
    for (auto pair : inputHandles)
    {
        m_InputHandleMap[pair.first] = m_WorkingMemDescriptorMap.at(pair.second).m_Outputs[0];
        m_ValidationMap[pair.first] = false;
    }

    // For every input we need to store all locations from which that input's ITensorHandle* is read.
    // So we can, at a later point, swap in and out the ITensorHandle* at that location.
    for (auto inputConnectionInfo : inputConnections)
    {
        WorkingMemDescriptor& workingMemDescriptor = m_WorkingMemDescriptors[inputConnectionInfo.m_DescriptorIndex];

        auto pos = workingMemDescriptor.m_Inputs.begin();
        // The difference_type of a vector can be unsigned int or signed int depending on the std implementation
        // This cast removes any conversion warnings
        pos += numeric_cast<std::vector<ITensorHandle*>::difference_type>(inputConnectionInfo.m_InputIndex);

        m_InputConnectionMap[inputConnectionInfo.m_LayerBindingId].push_back(pos);
    }
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
