//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Packet.hpp"

#include <cstdint>

namespace armnn
{

namespace profiling
{

#pragma once

class CommandHandlerFunctor
{
public:
    CommandHandlerFunctor(uint32_t packetId, uint32_t version) : m_PacketId(packetId), m_Version(version) {};

    uint32_t GetPacketId() const;
    uint32_t GetVersion()  const;

    virtual void operator()(const Packet& packet) {};

private:
    uint32_t m_PacketId;
    uint32_t m_Version;
};

} // namespace profiling

} // namespace armnn
