//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <client/include/ILocalPacketHandler.hpp>

#include <common/include/Packet.hpp>

#include <cstdint>

namespace arm
{

namespace pipe
{

class IProfilingConnection
{
public:
    virtual ~IProfilingConnection() {}

    virtual bool IsOpen() const = 0;

    virtual void Close() =  0;

    virtual bool WritePacket(const unsigned char* buffer, uint32_t length) = 0;

    virtual arm::pipe::Packet ReadPacket(uint32_t timeout) = 0;
};

} // namespace pipe

} // namespace arm
