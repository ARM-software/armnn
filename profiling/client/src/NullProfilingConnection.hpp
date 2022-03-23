//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IProfilingConnection.hpp"

#include <common/include/IgnoreUnused.hpp>

namespace arm
{

namespace pipe
{

class NullProfilingConnection : public IProfilingConnection
{
    virtual bool IsOpen() const override { return true; };

    virtual void Close() override {};

    virtual bool WritePacket(const unsigned char* buffer, uint32_t length) override
    {
        arm::pipe::IgnoreUnused(buffer);
        arm::pipe::IgnoreUnused(length);
        return true;
    };

    virtual Packet ReadPacket(uint32_t timeout) override
    {
        arm::pipe::IgnoreUnused(timeout);
        return Packet(0);
    }

};

} // namespace pipe

} // namespace arm
