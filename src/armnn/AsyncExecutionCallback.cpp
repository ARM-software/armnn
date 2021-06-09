//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <AsyncExecutionCallback.hpp>

namespace armnn
{

namespace experimental
{

void AsyncExecutionCallback::Notify(armnn::Status status, InferenceTimingPair timeTaken)
{
    {
        std::lock_guard<std::mutex> hold(m_Mutex);
        // store results and mark as notified
        m_Status    = status;
        m_StartTime = timeTaken.first;
        m_EndTime   = timeTaken.second;
        m_NotificationQueue.push(m_InferenceId);
    }
    m_Condition.notify_all();
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
    auto cb = std::make_unique<AsyncExecutionCallback>(m_NotificationQueue, m_Mutex, m_Condition);
    InferenceId id = cb->GetInferenceId();
    m_Callbacks.insert({id, std::move(cb)});

    return m_Callbacks.at(id);
}

std::shared_ptr<AsyncExecutionCallback> AsyncCallbackManager::GetNotifiedCallback()
{
    std::unique_lock<std::mutex> lock(m_Mutex);

    m_Condition.wait(lock, [this] { return !m_NotificationQueue.empty(); });

    InferenceId id = m_NotificationQueue.front();
    m_NotificationQueue.pop();

    std::shared_ptr<AsyncExecutionCallback> callback = m_Callbacks.at(id);
    m_Callbacks.erase(id);
    return callback;
}

} // namespace experimental

} // namespace armnn