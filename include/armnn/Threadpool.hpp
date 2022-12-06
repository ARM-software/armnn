//
// Copyright © 2021-2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#if !defined(ARMNN_DISABLE_THREADS)

#pragma once

#include "IRuntime.hpp"
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>
#include <stdint.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <queue>
#include <iosfwd>
#include <memory>
#include <tuple>
#include <vector>

namespace armnn
{
namespace experimental
{
class IAsyncExecutionCallback;
class IWorkingMemHandle;

class Threadpool
{
public:
    Threadpool(std::size_t numThreads,
               IRuntime* runtimePtr,
               std::vector<std::shared_ptr<IWorkingMemHandle>> memHandles);

    ~Threadpool()
    {
        TerminateThreadPool();
    }

    void LoadMemHandles(std::vector<std::shared_ptr<IWorkingMemHandle>> memHandles);
    void UnloadMemHandles(NetworkId networkId);

    /// Schedule an asynchronous execution on the loaded network
    void Schedule(NetworkId networkId,
                  const InputTensors &inputTensors,
                  const OutputTensors &outputTensors,
                  const QosExecPriority priority,
                  std::shared_ptr<IAsyncExecutionCallback> cb);

    void TerminateThreadPool() noexcept;

private:
    using ExecutionTuple = std::tuple<NetworkId,
                                      InputTensors,
                                      OutputTensors,
                                      std::shared_ptr<IAsyncExecutionCallback>>;

    using ExecutionQueue = std::queue<std::shared_ptr<ExecutionTuple>>;

    void ProcessExecPriorities(uint32_t index);

    IRuntime* m_RuntimePtr;

    ExecutionQueue m_HighPriorityQueue;
    ExecutionQueue m_MediumPriorityQueue;
    ExecutionQueue m_LowPriorityQueue;

    // Condition Variables require mutex which will guard the shared state.
    // Has an event happened? Stop signal for example
    std::condition_variable m_ThreadPoolEvent;
    std::mutex m_ThreadPoolMutex;

    // The shared state for conditional variable
    bool m_TerminatePool = false;

    std::unordered_map<NetworkId, std::vector<std::shared_ptr<IWorkingMemHandle>>> m_WorkingMemHandleMap;
    std::vector<std::unique_ptr<std::thread>> m_Threads;
};

} // namespace experimental

} // namespace armnn

#endif
