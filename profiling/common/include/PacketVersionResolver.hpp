//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "EncodeVersion.hpp"

namespace arm
{

namespace pipe
{

class PacketKey final
{
public:
    PacketKey(uint32_t familyId, uint32_t packetId) : m_FamilyId(familyId), m_PacketId(packetId) {}

    uint32_t GetFamilyId() { return m_FamilyId; }
    uint32_t GetPacketId() { return m_PacketId; }

    bool operator< (const PacketKey& rhs) const;
    bool operator> (const PacketKey& rhs) const;
    bool operator<=(const PacketKey& rhs) const;
    bool operator>=(const PacketKey& rhs) const;
    bool operator==(const PacketKey& rhs) const;
    bool operator!=(const PacketKey& rhs) const;

private:
    uint32_t m_FamilyId;
    uint32_t m_PacketId;
};

static const PacketKey ActivateTimeLinePacket(0 , 6);
static const PacketKey DeactivateTimeLinePacket(0 , 7);

class PacketVersionResolver final
{
public:
    PacketVersionResolver()  = default;
    ~PacketVersionResolver() = default;

    Version ResolvePacketVersion(uint32_t familyId, uint32_t packetId) const;
};

} // namespace pipe

} // namespace arm
