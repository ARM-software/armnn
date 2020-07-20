//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandHandlerKey.hpp"

namespace arm
{

namespace pipe
{

uint32_t CommandHandlerKey::GetFamilyId() const
{
    return m_FamilyId;
}

uint32_t CommandHandlerKey::GetPacketId() const
{
    return m_PacketId;
}

uint32_t CommandHandlerKey::GetVersion() const
{
    return m_Version;
}

bool CommandHandlerKey::operator<(const CommandHandlerKey& rhs) const
{
    bool result = true;
    if (m_FamilyId == rhs.m_FamilyId)
    {
        if (m_PacketId == rhs.m_PacketId)
        {
            result = m_Version < rhs.m_Version;
        }
        else if (m_PacketId > rhs.m_PacketId)
        {
            result = false;
        }
    }
    else if (m_FamilyId > rhs.m_FamilyId)
    {
        result = false;
    }
    return result;
}

bool CommandHandlerKey::operator>(const CommandHandlerKey& rhs) const
{
    return rhs < *this;
}

bool CommandHandlerKey::operator<=(const CommandHandlerKey& rhs) const
{
    return !(*this > rhs);
}

bool CommandHandlerKey::operator>=(const CommandHandlerKey& rhs) const
{
    return !(*this < rhs);
}

bool CommandHandlerKey::operator==(const CommandHandlerKey& rhs) const
{
    return m_FamilyId == rhs.m_FamilyId && m_PacketId == rhs.m_PacketId && m_Version == rhs.m_Version;
}

bool CommandHandlerKey::operator!=(const CommandHandlerKey& rhs) const
{
    return !(*this == rhs);
}

} // namespace pipe

} // namespace arm
