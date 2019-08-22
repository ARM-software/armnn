//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdint>

namespace armnn
{

namespace profiling
{

class CommandHandlerKey
{
public:
    CommandHandlerKey(uint32_t packetId, uint32_t version) : m_PacketId(packetId), m_Version(version) {};

    uint32_t GetPacketId() const;
    uint32_t GetVersion()  const;

    bool operator< (const CommandHandlerKey& rhs) const;
    bool operator> (const CommandHandlerKey& rhs) const;
    bool operator<=(const CommandHandlerKey& rhs) const;
    bool operator>=(const CommandHandlerKey& rhs) const;
    bool operator==(const CommandHandlerKey& rhs) const;
    bool operator!=(const CommandHandlerKey& rhs) const;

private:
    uint32_t m_PacketId;
    uint32_t m_Version;
};

} // namespace profiling

} // namespace armnn
