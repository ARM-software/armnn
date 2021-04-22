//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Types.hpp"

namespace armnn
{

namespace experimental
{

class IAsyncExecutionCallback;
using IAsyncExecutionCallbackPtr = std::shared_ptr<IAsyncExecutionCallback>;

class IAsyncExecutionCallback
{
public:
    virtual ~IAsyncExecutionCallback() {};

    // Notify the AsyncExecutionCallback object of the armnn execution status
    virtual void Notify(armnn::Status status, InferenceTimingPair timeTaken) = 0;

    // Block the calling thread until the AsyncExecutionCallback object allows it to proceed
    virtual void Wait() const = 0;

    // Retrieve the ArmNN Status from the AsyncExecutionCallback that has been notified
    virtual armnn::Status GetStatus() const = 0;

    // Retrieve the start time before executing the inference
    virtual HighResolutionClock GetStartTime() const = 0;

    // Retrieve the time after executing the inference
    virtual HighResolutionClock GetEndTime() const = 0;

};

} // experimental

} // namespace armnn
