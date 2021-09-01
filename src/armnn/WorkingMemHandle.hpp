//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Layer.hpp"
#include "Network.hpp"
#include "WorkingMemDescriptor.hpp"

#include <armnn/IWorkingMemHandle.hpp>
#include <armnn/Tensor.hpp>

#include <unordered_map>
#include <mutex>

namespace armnn
{

namespace experimental
{


class WorkingMemHandle final : public IWorkingMemHandle
{

public:
    struct InputConnectionInfo
    {
        LayerBindingId m_LayerBindingId;
        unsigned int m_DescriptorIndex;
        unsigned int m_InputIndex;
    };

    WorkingMemHandle(NetworkId networkId) : m_NetworkId(networkId){}

    WorkingMemHandle(NetworkId networkId,
                     std::vector<std::pair<LayerBindingId, LayerGuid>> inputHandles,
                     std::vector<InputConnectionInfo> inputConnections,
                     std::vector<WorkingMemDescriptor> workingMemDescriptors,
                     std::unordered_map<LayerGuid, WorkingMemDescriptor> workingMemDescriptorMap,
                     std::vector<std::shared_ptr<IMemoryManager>> memoryManagers,
                     std::unordered_map<LayerGuid, std::vector<std::unique_ptr<ITensorHandle> > > ownedTensorHandles);

    ~WorkingMemHandle()
    { Free(); }

    NetworkId GetNetworkId() override
    {
        return m_NetworkId;
    }

    /// Allocate the backing memory required for execution. If this is not called, then allocation will be
    /// deferred to execution time. The mutex must be locked.
    void Allocate() override;

    /// Free the backing memory required for execution. The mutex must be locked.
    void Free() override;

    /// IsAllocated returns true if the backing memory is currently allocated. The mutex must be locked.
    bool IsAllocated() override
    {
        return m_IsAllocated;
    }

    /// Get a mutex which can be used for synchronizing access to the WorkingMemHandle object.
    std::mutex& GetMutex() override
    {
        return m_Mutex;
    }

    /// Get the WorkingMemDescriptor for a Layer. The mutex must be locked.
    WorkingMemDescriptor& GetWorkingMemDescriptor(LayerGuid id) override
    {
        auto result = m_WorkingMemDescriptorMap.find(id);
        ARMNN_ASSERT(result != m_WorkingMemDescriptorMap.end());
        return result->second;
    }

    /// Get the WorkingMemDescriptor at an index. The WorkingMemDescriptors are stored in the same order as
    /// the Workloads in a topologically sorted graph. The mutex must be locked.
    WorkingMemDescriptor& GetWorkingMemDescriptorAt(unsigned int id) override
    {
        return m_WorkingMemDescriptors[id];
    }

    ITensorHandle* GetInputHandle(LayerBindingId layerBindingId) const
    {
        return m_InputHandleMap.at(layerBindingId);
    };

    const std::vector<std::vector<ITensorHandle*>::iterator>& GetInputConnections(LayerBindingId layerBindingId) const
    {
        return m_InputConnectionMap.at(layerBindingId);
    };

    std::unordered_map<LayerBindingId, bool> GetValidationMap() const
    {
        return m_ValidationMap;
    };

private:
    NetworkId m_NetworkId;
    std::shared_ptr<ProfilerImpl> m_Profiler;

    std::unordered_map<LayerBindingId, ITensorHandle*> m_InputHandleMap;
    std::unordered_map<LayerBindingId, std::vector<std::vector<ITensorHandle*>::iterator>> m_InputConnectionMap;

    std::vector<WorkingMemDescriptor> m_WorkingMemDescriptors;
    std::unordered_map<LayerGuid, WorkingMemDescriptor> m_WorkingMemDescriptorMap;

    // Vector of IMemoryManagers that manage the WorkingMemHandle's memory
    std::vector<std::shared_ptr<IMemoryManager>> m_MemoryManagers;
    // TensorHandles owned by this WorkingMemHandle
    // constant tensor's can be shared by multiple WorkingMemHandles and so will not be stored here
    std::unordered_map<LayerGuid, std::vector<std::unique_ptr<ITensorHandle> > >  m_OwnedTensorHandles;

    std::unordered_map<LayerBindingId, bool> m_ValidationMap;
    bool m_IsAllocated;
    std::mutex m_Mutex;
};

} // end experimental namespace

} // end armnn namespace
