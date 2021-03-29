//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "backendsCommon/CpuTensorHandle.hpp"
#include "WorkingMemHandle.hpp"
#include "Network.hpp"

namespace armnn
{

namespace experimental
{

WorkingMemHandle::WorkingMemHandle(std::vector<WorkingMemDescriptor> workingMemDescriptors,
                                   std::unordered_map<LayerGuid, WorkingMemDescriptor> workingMemDescriptorMap) :
    m_WorkingMemDescriptors(workingMemDescriptors),
    m_WorkingMemDescriptorMap(workingMemDescriptorMap),
    m_IsAllocated(false),
    m_Mutex()
{}

void WorkingMemHandle::FreeWorkingMemory()
{
    for (auto workingMemDescriptor : m_WorkingMemDescriptors)
    {
        for (auto input : workingMemDescriptor.m_Inputs)
        {
            if (input)
            {
                delete input;
                input = nullptr;
            }
        }
        for (auto output : workingMemDescriptor.m_Outputs)
        {
            if (output)
            {
                delete output;
                output = nullptr;
            }
        }
    }
}

} // end experimental namespace

} // end armnn namespace
