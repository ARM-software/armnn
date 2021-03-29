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

namespace armnn
{

namespace experimental
{

class WorkingMemHandle final : public IWorkingMemHandle
{

public:
    WorkingMemHandle(std::vector<WorkingMemDescriptor> workingMemDescriptors,
                     std::unordered_map<LayerGuid, WorkingMemDescriptor> workingMemDescriptorMap);

    ~WorkingMemHandle() { FreeWorkingMemory(); }

    /// Allocate the backing memory required for execution. If this is not called, then allocation will be
    /// deferred to execution time. The mutex must be locked.
    void Allocate() override
    {
        if (m_IsAllocated)
        {
            return;
        }
        m_IsAllocated = true;

        // Iterate through all WorkingMemDescriptors calling allocate() on each input and output in turn
        for (auto workingMemDescriptor :  m_WorkingMemDescriptors)
        {
            for (auto& input : workingMemDescriptor.m_Inputs)
            {
                input->Allocate();
            }
            for (auto& output : workingMemDescriptor.m_Outputs)
            {
                output->Allocate();
            }
        }
    }

    /// Free the backing memory required for execution. The mutex must be locked.
    void Free() override
    {
        if (!m_IsAllocated)
        {
            return;
        }
        m_IsAllocated = false;

        // Iterate through all WorkingMemDescriptors calling free() on each input and output in turn
        for (auto workingMemDescriptor :  m_WorkingMemDescriptors)
        {
            for (auto& input : workingMemDescriptor.m_Inputs)
            {
                input->Unmap();
            }
            for (auto& output : workingMemDescriptor.m_Outputs)
            {
                output->Unmap();
            }
        }
    }

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

private:
    void FreeWorkingMemory();

    std::shared_ptr<ProfilerImpl> m_Profiler;

    std::vector<WorkingMemDescriptor> m_WorkingMemDescriptors;
    std::unordered_map<LayerGuid, WorkingMemDescriptor> m_WorkingMemDescriptorMap;
    bool m_IsAllocated;
    std::mutex m_Mutex;
};

} // end experimental namespace

} // end armnn namespace
