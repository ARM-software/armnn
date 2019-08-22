//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandHandlerFunctor.hpp"

namespace armnn
{

namespace profiling
{

uint32_t CommandHandlerFunctor::GetPacketId() const
{
    return m_PacketId;
}

uint32_t CommandHandlerFunctor::GetVersion() const
{
    return m_Version;
}

} // namespace profiling

} // namespace armnn
