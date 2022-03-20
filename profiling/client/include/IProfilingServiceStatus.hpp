//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ILocalPacketHandler.hpp"

#include <common/include/Packet.hpp>

#include <cstdint>

namespace arm
{

namespace pipe
{

class IProfilingServiceStatus
{
public:
    virtual void NotifyProfilingServiceActive() = 0;
    virtual void WaitForProfilingServiceActivation(unsigned int timeout) = 0;
    virtual ~IProfilingServiceStatus() {};
};

} // namespace pipe

} // namespace arm
