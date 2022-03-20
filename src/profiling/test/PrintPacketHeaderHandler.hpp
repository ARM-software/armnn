//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <client/include/ILocalPacketHandler.hpp>

#include <common/include/Packet.hpp>

namespace arm
{

namespace pipe
{

class PrintPacketHeaderHandler : public ILocalPacketHandler
{
    virtual std::vector<uint32_t> GetHeadersAccepted();

    virtual void HandlePacket(const arm::pipe::Packet& packet);
};

} // namespace pipe

} // namespace arm
