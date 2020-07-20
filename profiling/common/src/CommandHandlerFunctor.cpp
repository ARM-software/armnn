//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandHandlerFunctor.hpp"

namespace arm
{

namespace pipe
{

uint32_t CommandHandlerFunctor::GetFamilyId() const
{
    return m_FamilyId;
}

uint32_t CommandHandlerFunctor::GetPacketId() const
{
    return m_PacketId;
}

uint32_t CommandHandlerFunctor::GetVersion() const
{
    return m_Version;
}

} // namespace pipe

} // namespace arm
