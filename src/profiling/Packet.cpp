//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Packet.hpp"

namespace armnn
{

namespace profiling
{

std::uint32_t Packet::GetHeader() const
{
    return m_Header;
}

std::uint32_t Packet::GetPacketFamily() const
{
    return m_PacketFamily;
}

std::uint32_t Packet::GetPacketId() const
{
    return m_PacketId;
}

std::uint32_t Packet::GetLength() const
{
    return m_Length;
}

const char* Packet::GetData()
{
    return m_Data;
}

std::uint32_t Packet::GetPacketClass() const
{
    return (m_PacketId >> 3);
}

std::uint32_t Packet::GetPacketType() const
{
    return (m_PacketId & 7);
}

} // namespace profiling

} // namespace armnn
