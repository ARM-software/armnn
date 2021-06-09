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
};

} // experimental

} // namespace armnn
