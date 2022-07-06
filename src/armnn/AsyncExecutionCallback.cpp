//
// Copyright © 2021-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <AsyncExecutionCallback.hpp>

namespace armnn
{

namespace experimental
{

InferenceId AsyncExecutionCallback::nextID = 0u;

void AsyncExecutionCallback::Notify(armnn::Status status, InferenceTimingPair timeTaken)
{
    {
#if !defined(ARMNN_DISABLE_THREADS)
        std::lock_guard<std::mutex> hold(m_Mutex);
#endif
        // store results and mark as notified
        m_Status    = status;
        m_StartTime = timeTaken.first;
        m_EndTime   = timeTaken.second;
        m_NotificationQueue.push(m_InferenceId);
    }
#if !defined(ARMNN_DISABLE_THREADS)
    m_Condition.notify_all();
#endif
}

armnn::Status AsyncExecutionCallback::GetStatus() const
{
    return m_Status;
}

HighResolutionClock AsyncExecutionCallback::GetStartTime() const
{
    return m_StartTime;
}

HighResolutionClock AsyncExecutionCallback::GetEndTime() const
{
    return m_EndTime;
}

std::shared_ptr<AsyncExecutionCallback> AsyncCallbackManager::GetNewCallback()
{
    auto cb = std::make_unique<AsyncExecutionCallback>(m_NotificationQueue
#if !defined(ARMNN_DISABLE_THREADS)
                                                       , m_Mutex
                                                       , m_Condition
#endif
        );
    InferenceId id = cb->GetInferenceId();
    m_Callbacks.insert({id, std::move(cb)});

    return m_Callbacks.at(id);
}

std::shared_ptr<AsyncExecutionCallback> AsyncCallbackManager::GetNotifiedCallback()
{
#if !defined(ARMNN_DISABLE_THREADS)
    std::unique_lock<std::mutex> lock(m_Mutex);

    m_Condition.wait(lock, [this] { return !m_NotificationQueue.empty(); });
#endif
    InferenceId id = m_NotificationQueue.front();
    m_NotificationQueue.pop();

    std::shared_ptr<AsyncExecutionCallback> callback = m_Callbacks.at(id);
    m_Callbacks.erase(id);
    return callback;
}

} // namespace experimental

} // namespace armnn
