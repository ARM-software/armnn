//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IProfilingConnection.hpp"

#include <armnn/utility/IgnoreUnused.hpp>

namespace armnn
{

namespace profiling
{

class NullProfilingConnection : public IProfilingConnection
{
    virtual bool IsOpen() const override { return true; };

    virtual void Close() override {};

    virtual bool WritePacket(const unsigned char* buffer, uint32_t length) override
    {
        armnn::IgnoreUnused(buffer);
        armnn::IgnoreUnused(length);
        return true;
    };

    virtual Packet ReadPacket(uint32_t timeout) override
    {
        armnn::IgnoreUnused(timeout);
        return Packet(0);
    }

};

} // namespace profiling

} // namespace armnn