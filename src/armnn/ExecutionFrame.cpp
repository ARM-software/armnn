//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ExecutionFrame.hpp"

using namespace std;

namespace armnn
{
ExecutionFrame::ExecutionFrame() {}

IExecutionFrame* ExecutionFrame::ExecuteWorkloads(IExecutionFrame* previousFrame)
{
    IgnoreUnused(previousFrame);
    for (auto& workload: m_WorkloadQueue)
    {
        workload->Execute();
    }
    return m_NextExecutionFrame;
}

void ExecutionFrame::PostAllocationConfigure()
{
    for (auto&& workloadPtr: m_WorkloadQueue)
    {
        workloadPtr.get()->PostAllocationConfigure();
    }
}

void ExecutionFrame::RegisterDebugCallback(const DebugCallbackFunction& func)
{
    for (auto&& workloadPtr: m_WorkloadQueue)
    {
        workloadPtr.get()->RegisterDebugCallback(func);
    }
}

void ExecutionFrame::AddWorkloadToQueue(std::unique_ptr<IWorkload> workload)
{
    m_WorkloadQueue.push_back(move(workload));
}

void ExecutionFrame::SetNextExecutionFrame(IExecutionFrame* nextExecutionFrame)
{
    m_NextExecutionFrame = nextExecutionFrame;
}

}