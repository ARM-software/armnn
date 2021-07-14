//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//


#include <armnn/Threadpool.hpp>

#include <armnn/utility/Timer.hpp>

namespace armnn
{
namespace experimental
{

Threadpool::Threadpool(std::size_t numThreads,
                       IRuntime* runtimePtr,
                       std::vector<std::shared_ptr<IWorkingMemHandle>> memHandles)
    : m_RuntimePtr(runtimePtr)
{
    for (auto i = 0u; i < numThreads; ++i)
    {
        m_Threads.emplace_back(std::make_unique<std::thread>(&Threadpool::ProcessExecPriorities, this, i));
    }

    LoadMemHandles(memHandles);
}

void Threadpool::LoadMemHandles(std::vector<std::shared_ptr<IWorkingMemHandle>> memHandles)
{
    if (memHandles.size() == 0)
    {
        throw armnn::RuntimeException("Threadpool::UnloadMemHandles: Size of memHandles vector must be greater than 0");
    }

    if (memHandles.size() != m_Threads.size())
    {
        throw armnn::RuntimeException(
                "Threadpool::UnloadMemHandles: Size of memHandles vector must match the number of threads");
    }

    NetworkId networkId = memHandles[0]->GetNetworkId();
    for (uint32_t i = 1; i < memHandles.size(); ++i)
    {
        if (networkId != memHandles[i]->GetNetworkId())
        {
            throw armnn::RuntimeException(
                    "Threadpool::UnloadMemHandles: All network ids must be identical in memHandles");
        }
    }

    std::pair<NetworkId, std::vector<std::shared_ptr<IWorkingMemHandle>>> pair {networkId, memHandles};

    m_WorkingMemHandleMap.insert(pair);
}

void Threadpool::UnloadMemHandles(NetworkId networkId)
{
    if (m_WorkingMemHandleMap.find(networkId) != m_WorkingMemHandleMap.end())
    {
        m_WorkingMemHandleMap.erase(networkId);
    }
    else
    {
       throw armnn::RuntimeException("Threadpool::UnloadMemHandles: Unknown NetworkId");
    }
}

void Threadpool::Schedule(NetworkId networkId,
                          const InputTensors& inputTensors,
                          const OutputTensors& outputTensors,
                          const QosExecPriority priority,
                          std::shared_ptr<IAsyncExecutionCallback> cb)
{
    if (m_WorkingMemHandleMap.find(networkId) == m_WorkingMemHandleMap.end())
    {
        throw armnn::RuntimeException("Threadpool::UnloadMemHandles: Unknown NetworkId");
    }

    // Group execution parameters so that they can be easily added to the queue
    ExecutionTuple groupExecParams = std::make_tuple(networkId, inputTensors, outputTensors, cb);

    std::shared_ptr<ExecutionTuple> operation = std::make_shared<ExecutionTuple>(groupExecParams);

    // Add a message to the queue and notify the request thread
    std::unique_lock<std::mutex> lock(m_ThreadPoolMutex);
    switch (priority)
    {
        case QosExecPriority::High:
            m_HighPriorityQueue.push(operation);
            break;
        case QosExecPriority::Low:
            m_LowPriorityQueue.push(operation);
            break;
        case QosExecPriority::Medium:
        default:
            m_MediumPriorityQueue.push(operation);
    }
    m_ThreadPoolEvent.notify_one();
}

void Threadpool::TerminateThreadPool() noexcept
{
    {
        std::unique_lock<std::mutex> threadPoolLock(m_ThreadPoolMutex);
        m_TerminatePool = true;
    }

    m_ThreadPoolEvent.notify_all();

    for (auto &thread : m_Threads)
    {
        thread->join();
    }
}

void Threadpool::ProcessExecPriorities(uint32_t index)
{
    int expireRate = EXPIRE_RATE;
    int highPriorityCount = 0;
    int mediumPriorityCount = 0;

    while (true)
    {
        std::shared_ptr<ExecutionTuple> currentExecInProgress(nullptr);
        {
            // Wait for a message to be added to the queue
            // This is in a separate scope to minimise the lifetime of the lock
            std::unique_lock<std::mutex> lock(m_ThreadPoolMutex);

            m_ThreadPoolEvent.wait(lock,
                                   [=]
                                   {
                                       return m_TerminatePool || !m_HighPriorityQueue.empty() ||
                                              !m_MediumPriorityQueue.empty() || !m_LowPriorityQueue.empty();
                                   });

            if (m_TerminatePool && m_HighPriorityQueue.empty() && m_MediumPriorityQueue.empty() &&
                m_LowPriorityQueue.empty())
            {
                break;
            }

            // Get the message to process from the front of each queue based on priority from high to low
            // Get high priority first if it does not exceed the expire rate
            if (!m_HighPriorityQueue.empty() && highPriorityCount < expireRate)
            {
                currentExecInProgress = m_HighPriorityQueue.front();
                m_HighPriorityQueue.pop();
                highPriorityCount += 1;
            }
                // If high priority queue is empty or the count exceeds the expire rate, get medium priority message
            else if (!m_MediumPriorityQueue.empty() && mediumPriorityCount < expireRate)
            {
                currentExecInProgress = m_MediumPriorityQueue.front();
                m_MediumPriorityQueue.pop();
                mediumPriorityCount += 1;
                // Reset high priority count
                highPriorityCount = 0;
            }
                // If medium priority queue is empty or the count exceeds the expire rate, get low priority message
            else if (!m_LowPriorityQueue.empty())
            {
                currentExecInProgress = m_LowPriorityQueue.front();
                m_LowPriorityQueue.pop();
                // Reset high and medium priority count
                highPriorityCount = 0;
                mediumPriorityCount = 0;
            }
            else
            {
                // Reset high and medium priority count
                highPriorityCount = 0;
                mediumPriorityCount = 0;
                continue;
            }
        }

        // invoke the asynchronous execution method
        auto networkId = std::get<0>(*currentExecInProgress);
        auto inputTensors = std::get<1>(*currentExecInProgress);
        auto outputTensors = std::get<2>(*currentExecInProgress);
        auto cb = std::get<3>(*currentExecInProgress);

        // Get time at start of inference
        HighResolutionClock startTime = armnn::GetTimeNow();

        try // executing the inference
        {
            IWorkingMemHandle& memHandle = *(m_WorkingMemHandleMap.at(networkId))[index];

            // Execute and populate the time at end of inference in the callback
            m_RuntimePtr->Execute(memHandle, inputTensors, outputTensors) == Status::Success ?
            cb->Notify(Status::Success, std::make_pair(startTime, armnn::GetTimeNow())) :
            cb->Notify(Status::Failure, std::make_pair(startTime, armnn::GetTimeNow()));
        }
        catch (const RuntimeException&)
        {
            cb->Notify(Status::Failure, std::make_pair(startTime, armnn::GetTimeNow()));
        }
    }
}

} // namespace experimental

} // namespace armnn
