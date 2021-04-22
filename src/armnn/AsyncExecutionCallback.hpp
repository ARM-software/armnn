//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/IAsyncExecutionCallback.hpp>
#include <armnn/Types.hpp>
#include <condition_variable>

#include <mutex>
#include <thread>

namespace armnn
{

namespace experimental
{

class AsyncExecutionCallback final : public IAsyncExecutionCallback
{
public:
    AsyncExecutionCallback()
    {}
    ~AsyncExecutionCallback()
    {}

    void Notify(armnn::Status status, InferenceTimingPair timeTaken);
    void Wait() const;

    armnn::Status GetStatus() const;
    HighResolutionClock GetStartTime() const;
    HighResolutionClock GetEndTime() const;

private:
    mutable std::mutex              m_Mutex;
    mutable std::condition_variable m_Condition;

    HighResolutionClock m_StartTime;
    HighResolutionClock m_EndTime;
    armnn::Status       m_Status   = Status::Failure;
    bool                m_Notified = false;
};

} // namespace experimental

} // namespace armnn