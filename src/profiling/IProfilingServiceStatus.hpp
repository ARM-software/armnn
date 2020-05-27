//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <Packet.hpp>
#include <armnn/profiling/ILocalPacketHandler.hpp>

#include <cstdint>

namespace armnn
{

namespace profiling
{

class IProfilingServiceStatus
{
public:
    virtual void NotifyProfilingServiceActive() = 0;
    virtual void WaitForProfilingServiceActivation(unsigned int timeout) = 0;
    virtual ~IProfilingServiceStatus() {};
};

} // namespace profiling

} // namespace armnn
