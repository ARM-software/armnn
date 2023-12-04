//
// Copyright © 2019-2021,2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/Workload.hpp>

namespace armnn
{

using WorkloadQueue = std::vector< std::unique_ptr<IWorkload> >;

/// ExecutionFrame interface to enqueue a workload computation.
class IExecutionFrame
{

public:
    virtual ~IExecutionFrame() {}

    virtual IExecutionFrame* ExecuteWorkloads(IExecutionFrame* previousFrame) = 0;
    virtual void PostAllocationConfigure() {};
    virtual void RegisterDebugCallback(const DebugCallbackFunction&) {};
};

class ExecutionFrame: public IExecutionFrame
{
public:
    ExecutionFrame();

    IExecutionFrame* ExecuteWorkloads(IExecutionFrame* previousFrame) override ;
    void PostAllocationConfigure() override;
    void RegisterDebugCallback(const DebugCallbackFunction& func) override ;
    void AddWorkloadToQueue(std::unique_ptr<IWorkload> workload);
    void SetNextExecutionFrame(IExecutionFrame* nextExecutionFrame);
private:
    WorkloadQueue m_WorkloadQueue;
    IExecutionFrame* m_NextExecutionFrame = nullptr;
};

}
