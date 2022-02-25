//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace arm
{

namespace pipe
{

class IConsumer
{
public:
    virtual ~IConsumer() {}

    /// Set a "ready to read" flag in the buffer to notify the reading thread to start reading it.
    virtual void SetReadyToRead() = 0;
};

} // namespace pipe

} // namespace arm

