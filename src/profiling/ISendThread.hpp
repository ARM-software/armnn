//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IProfilingConnection.hpp"

namespace armnn
{

namespace profiling
{

class ISendThread
{
public:
    virtual ~ISendThread() {}

    /// Start the thread
    virtual void Start(IProfilingConnection& profilingConnection) = 0;

    /// Stop the thread
    virtual void Stop(bool rethrowSendThreadExceptions = true) = 0;
};

} // namespace profiling

} // namespace armnn

