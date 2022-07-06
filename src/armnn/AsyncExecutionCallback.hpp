//
// Copyright © 2021-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/IAsyncExecutionCallback.hpp>
#include <armnn/IWorkingMemHandle.hpp>
#include <armnn/Types.hpp>

#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>
#include <unordered_map>

namespace armnn
{

namespace experimental
{

using InferenceId = uint64_t;
class AsyncExecutionCallback final : public IAsyncExecutionCallback
{
private:
    static InferenceId nextID;

public:
    AsyncExecutionCallback(std::queue<InferenceId>& notificationQueue
#if !defined(ARMNN_DISABLE_THREADS)
                           , std::mutex& mutex
                           , std::condition_variable& condition
#endif
        )
        : m_NotificationQueue(notificationQueue)
#if !defined(ARMNN_DISABLE_THREADS)
        , m_Mutex(mutex)
        , m_Condition(condition)
#endif
        , m_InferenceId(++nextID)
    {}

    ~AsyncExecutionCallback()
    {}

    void Notify(armnn::Status status, InferenceTimingPair timeTaken);

    InferenceId GetInferenceId()
    {
        return m_InferenceId;
    }

    armnn::Status GetStatus() const;
    HighResolutionClock GetStartTime() const;
    HighResolutionClock GetEndTime() const;

private:
    std::queue<InferenceId>& m_NotificationQueue;
#if !defined(ARMNN_DISABLE_THREADS)
    std::mutex&              m_Mutex;
    std::condition_variable& m_Condition;
#endif

    HighResolutionClock m_StartTime;
    HighResolutionClock m_EndTime;
    armnn::Status       m_Status = Status::Failure;
    InferenceId m_InferenceId;
};

// Manager to create and monitor AsyncExecutionCallbacks
// GetNewCallback will create a callback for use in Threadpool::Schedule
// GetNotifiedCallback will return the first callback to be notified (finished execution)
class AsyncCallbackManager
{
public:
    std::shared_ptr<AsyncExecutionCallback> GetNewCallback();
    std::shared_ptr<AsyncExecutionCallback> GetNotifiedCallback();

private:
#if !defined(ARMNN_DISABLE_THREADS)
    std::mutex              m_Mutex;
    std::condition_variable m_Condition;
#endif
    std::unordered_map<InferenceId, std::shared_ptr<AsyncExecutionCallback>> m_Callbacks;
    std::queue<InferenceId> m_NotificationQueue;
};

} // namespace experimental

} // namespace armnn
