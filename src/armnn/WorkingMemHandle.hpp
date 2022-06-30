//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ExecutionData.hpp"
#include "Layer.hpp"
#include "Network.hpp"
#include "WorkingMemDescriptor.hpp"

#include <armnn/IWorkingMemHandle.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/backends/IBackendInternal.hpp>

#include <unordered_map>
#include <mutex>
#include <backendsCommon/MemoryManager.hpp>

namespace armnn
{

namespace experimental
{

using BackendPtrMap = std::unordered_map<BackendId, IBackendInternalUniquePtr>;

class WorkingMemHandle final : public IWorkingMemHandle
{

public:
    struct InputMemDescriptorCoords
    {
        LayerBindingId m_LayerBindingId;

        std::vector<std::pair<unsigned int, unsigned int>> m_InputSlotCoords;
    };

    struct OutputMemDescriptorCoords
    {
        std::vector<LayerBindingId> m_LayerBindingIds;

        std::pair<unsigned int, unsigned int> m_OutputSlotCoords;
        std::vector<std::pair<unsigned int, unsigned int>> m_InputSlotCoords;
    };

    WorkingMemHandle(NetworkId networkId) : m_NetworkId(networkId){}

    WorkingMemHandle(NetworkId networkId,
                     std::vector<InputMemDescriptorCoords> inputLayerInfo,
                     std::vector<OutputMemDescriptorCoords> outputLayerInfo,
                     std::vector<WorkingMemDescriptor> workingMemDescriptors,
                     std::unique_ptr<MemoryManager> memoryManager,
                     std::vector<std::pair<std::shared_ptr<TensorMemory>, MemorySource>> tensorMemory,
                     std::vector<std::unique_ptr<ITensorHandle>> managedTensorHandles,
                     std::vector<std::unique_ptr<ITensorHandle>> unmanagedTensorHandles,
                     std::vector<std::pair<BackendId, ExecutionData>> executionDataVec,
                     BackendPtrMap* backends);

    ~WorkingMemHandle()
    { Free(); }

    NetworkId GetNetworkId() override
    {
        return m_NetworkId;
    }

    /// Allocate the backing memory required for execution. If this is not called, then allocation will be
    /// deferred to execution time.
    void Allocate() override;

    /// Free the backing memory required for execution.
    void Free() override;

    /// IsAllocated returns true if the backing memory is currently allocated.
    bool IsAllocated() override
    {
        return m_IsAllocated;
    }

    /// Get the WorkingMemDescriptor at an index. The WorkingMemDescriptors are stored in the same order as
    /// the Workloads in a topologically sorted graph.
    WorkingMemDescriptor& GetWorkingMemDescriptorAt(unsigned int id) override
    {
        return m_WorkingMemDescriptors[id];
    }

    /// Get the ExecutionData at an index.
    /// The ExecutionData is paired with a BackendId to be able to call backend specific functions upon it.
    /// The ExecutionData are stored in the same order as the Workloads in a topologically sorted graph.
    std::pair<BackendId, ExecutionData>& GetExecutionDataAt(unsigned int id) override
    {
        return m_ExecutionDataVec[id];
    }

    ITensorHandle* GetInputHandle(LayerBindingId layerBindingId) const
    {
        return m_InputHandleMap.at(layerBindingId);
    };

    ITensorHandle* GetOutputHandle(LayerBindingId layerBindingId) const
    {
        return m_OutputHandleMap.at(layerBindingId);
    };

    const std::vector<std::vector<ITensorHandle*>::iterator>& GetInputConnections(LayerBindingId layerBindingId) const
    {
        return m_InputConnectionMap.at(layerBindingId);
    };

    const std::vector<std::vector<ITensorHandle*>::iterator>& GetOutputConnection(LayerBindingId layerBindingId) const
    {
        return m_OutputConnectionMap.at(layerBindingId);
    };

    void MemSyncOutputs();

    std::vector<LayerBindingId>& GetBindingIdVector()
    {
        return m_BindingIdVec;
    };

    void ValidateBindingIds();

private:
    using DifferenceType = std::vector<ITensorHandle*>::difference_type;
    NetworkId m_NetworkId;

    std::unordered_map<LayerBindingId, ITensorHandle*> m_InputHandleMap;
    std::unordered_map<LayerBindingId, ITensorHandle*> m_OutputHandleMap;
    std::unordered_map<LayerBindingId, std::vector<std::vector<ITensorHandle*>::iterator>> m_InputConnectionMap;
    std::unordered_map<LayerBindingId, std::vector<std::vector<ITensorHandle*>::iterator>> m_OutputConnectionMap;

    std::vector<WorkingMemDescriptor> m_WorkingMemDescriptors;

    std::unique_ptr<MemoryManager> m_MemoryManager;

    // Memory to be imported into the tensorHandles after allocation
    std::vector<std::pair<std::shared_ptr<TensorMemory>, MemorySource>> m_TensorMemory;

    // Tensors that will need to be allocated internally within armnn
    std::vector<std::unique_ptr<ITensorHandle>> m_ManagedTensorHandles;

    // Tensors that will be allocated externally by the user
    std::vector<std::unique_ptr<ITensorHandle>> m_UnmanagedTensorHandles;

    std::unordered_map<LayerBindingId, bool> m_InputValidationMap;
    std::unordered_map<LayerBindingId, bool> m_OutputValidationMap;

    std::vector<LayerBindingId> m_BindingIdVec;

    DifferenceType m_InputSize;

    bool m_IsAllocated;

    std::vector<std::pair<BackendId, ExecutionData>> m_ExecutionDataVec;

    BackendPtrMap* m_Backends;
};

} // end experimental namespace

} // end armnn namespace
