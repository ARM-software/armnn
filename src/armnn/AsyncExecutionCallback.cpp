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
        if (m_Notified)
        {
            return;
        }
        // store results and mark as notified
        m_Status    = status;
        m_StartTime = timeTaken.first;
        m_EndTime   = timeTaken.second;
        m_Notified  = true;
    }
    m_Condition.notify_all();
}

void AsyncExecutionCallback::Wait() const
{
    std::unique_lock<std::mutex> lock(m_Mutex);
    m_Condition.wait(lock, [this] { return m_Notified; });
}

armnn::Status AsyncExecutionCallback::GetStatus() const
{
    Wait();
    return m_Status;
}

HighResolutionClock AsyncExecutionCallback::GetStartTime() const
{
    Wait();
    return m_StartTime;
}

HighResolutionClock AsyncExecutionCallback::GetEndTime() const
{
    Wait();
    return m_EndTime;
}

} // namespace experimental

} // namespace armnn