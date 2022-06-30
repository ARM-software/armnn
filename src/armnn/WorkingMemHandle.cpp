//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "WorkingMemHandle.hpp"
#include "Network.hpp"
#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <fmt/format.h>

namespace armnn
{

namespace experimental
{

WorkingMemHandle::WorkingMemHandle(NetworkId networkId,
        std::vector<InputMemDescriptorCoords> inputLayerInfo,
        std::vector<OutputMemDescriptorCoords> outputLayerInfo,
        std::vector<WorkingMemDescriptor> workingMemDescriptors,
        std::unique_ptr<MemoryManager> memoryManager,
        std::vector<std::pair<std::shared_ptr<TensorMemory>, MemorySource>> tensorMemory,
        std::vector<std::unique_ptr<ITensorHandle>> managedTensorHandles,
        std::vector<std::unique_ptr<ITensorHandle>> unmanagedTensorHandles,
        std::vector<std::pair<BackendId, ExecutionData>> executionDataVec,
        BackendPtrMap* backends)
    : m_NetworkId(networkId)
    , m_WorkingMemDescriptors(workingMemDescriptors)
    , m_MemoryManager(std::move(memoryManager))
    , m_TensorMemory(std::move(tensorMemory))
    , m_ManagedTensorHandles(std::move(managedTensorHandles))
    , m_UnmanagedTensorHandles(std::move(unmanagedTensorHandles))
    , m_InputSize(numeric_cast<DifferenceType>(inputLayerInfo.size()))
    , m_IsAllocated(false)
    , m_ExecutionDataVec(executionDataVec)
    , m_Backends(backends)
{
    for (const auto& inputInfo : inputLayerInfo)
    {
        m_InputValidationMap[inputInfo.m_LayerBindingId] = false;

        // Map the LayerBindingIds to the corresponding input ITensorHandle*
        auto memDesc = m_WorkingMemDescriptors.at(inputInfo.m_InputSlotCoords[0].first);
        ITensorHandle* inputTensorHandle = memDesc.m_Inputs[inputInfo.m_InputSlotCoords[0].second];
        m_InputHandleMap[inputInfo.m_LayerBindingId] = inputTensorHandle;

        // For every input we need to store all locations from which that input's ITensorHandle* is read.
        // So we can, at a later point, swap in and out the ITensorHandle* at that location.
        for (auto inputSlot : inputInfo.m_InputSlotCoords)
        {
            WorkingMemDescriptor& workingMemDescriptor = m_WorkingMemDescriptors.at(inputSlot.first);

            auto inputPos = workingMemDescriptor.m_Inputs.begin();

            // The DifferenceType of a vector can be unsigned int or signed int depending on the std implementation
            // This cast removes any conversion warnings
            inputPos += numeric_cast<DifferenceType>(inputSlot.second);
            m_InputConnectionMap[inputInfo.m_LayerBindingId].push_back(inputPos);
        }
    }
    size_t bindingIdCount = inputLayerInfo.size();
    for (const auto& outputInfo : outputLayerInfo)
    {
        for (auto bindingId : outputInfo.m_LayerBindingIds)
        {
            m_OutputValidationMap[bindingId] = false;

            // Store the outputSlot position of the tensorhandle
            auto outputPos = m_WorkingMemDescriptors.at(outputInfo.m_OutputSlotCoords.first).m_Outputs.begin();
            outputPos += numeric_cast<DifferenceType>(outputInfo.m_OutputSlotCoords.second);

            m_OutputHandleMap[bindingId] = *outputPos;
        }
        bindingIdCount += outputInfo.m_LayerBindingIds.size();
        // More than one layerBinding id means the tensorhandle is connected to more than one OutputLayer.
        // Importing in this case would likely cause unexpected behaviour, so we disallow it.
        if (outputInfo.m_LayerBindingIds.size() != 1)
        {
            continue;
        }

        // Store the inputSlot positions of the tensorhandle
        for (auto outputSlot : outputInfo.m_InputSlotCoords)
        {
            WorkingMemDescriptor& workingMemDescriptor = m_WorkingMemDescriptors.at(outputSlot.first);

            auto inputPos = workingMemDescriptor.m_Inputs.begin();

            // The DifferenceType of a vector can be unsigned int or signed int depending on the std implementation
            // This cast removes any conversion warnings
            inputPos += numeric_cast<DifferenceType>(outputSlot.second);
            m_OutputConnectionMap[outputInfo.m_LayerBindingIds[0]].push_back(inputPos);
        }
    }
    m_BindingIdVec = std::vector<LayerBindingId>(bindingIdCount);
    IgnoreUnused(m_UnmanagedTensorHandles);
}

void WorkingMemHandle::Allocate()
{
    if (m_IsAllocated)
    {
        return;
    }
    m_IsAllocated = true;

    m_MemoryManager->Allocate();

    for (unsigned int i = 0; i < m_TensorMemory.size(); ++i)
    {
        m_ManagedTensorHandles[i]->Import(m_TensorMemory[i].first->m_Data, m_TensorMemory[i].second);
    }

    // Assign previously allocated ExecutionData. Needs to be assigned after allocation so the void* are allocated.
    for (unsigned int i = 0; i < m_ExecutionDataVec.size(); ++i)
    {
        auto& backend = m_Backends->at(m_ExecutionDataVec[i].first);

        ExecutionData executionData = backend->CreateExecutionData(GetWorkingMemDescriptorAt(i));
        m_ExecutionDataVec[i].second = executionData;
    }
}

void WorkingMemHandle::Free()
{
    if (!m_IsAllocated)
    {
        return;
    }
    m_IsAllocated = false;

    m_MemoryManager->Deallocate();
}

void WorkingMemHandle::MemSyncOutputs()
{
    for (auto output : m_OutputConnectionMap)
    {
        (*output.second[0])->Map(true);
        (*output.second[0])->Unmap();
    }
}

void WorkingMemHandle::ValidateBindingIds()
{
    auto resetInputValidationMap = [&]()
    {
        for (auto& pair: m_InputValidationMap)
        {
            pair.second = false;
        }
    };

    auto resetOutputValidationMap = [&]()
    {
        for (auto& pair: m_OutputValidationMap)
        {
            pair.second = false;
        }
    };

    std::for_each(m_BindingIdVec.begin(), m_BindingIdVec.begin() + m_InputSize, [&](LayerBindingId id)
    {
        try
        {
            bool& isUsed = m_InputValidationMap.at(id);
            if (isUsed)
            {
                resetInputValidationMap();
                throw InvalidArgumentException(fmt::format("Duplicate Input LayerBindingId: {}", id));
            }
            isUsed = true;
        }
        catch (const std::out_of_range&)
        {
            resetInputValidationMap();
            throw InvalidArgumentException(fmt::format("Unknown Input LayerBindingId: {}", id));
        }
    });
    resetInputValidationMap();

    std::for_each(m_BindingIdVec.begin() + m_InputSize, m_BindingIdVec.end(), [&](LayerBindingId id)
    {
        try
        {
            bool& isUsed = m_OutputValidationMap.at(id);
            if (isUsed)
            {
                resetOutputValidationMap();
                throw InvalidArgumentException(fmt::format("Duplicate Output LayerBindingId: {}", id));
            }
            isUsed = true;
        }
        catch (const std::out_of_range&)
        {
            resetOutputValidationMap();
            throw InvalidArgumentException(fmt::format("Unknown Output LayerBindingId: {}", id));
        }
    });
    resetOutputValidationMap();
}

} // end experimental namespace

} // end armnn namespace
