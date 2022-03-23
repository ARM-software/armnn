//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TimelinePacketWriterFactory.hpp"

#include "SendTimelinePacket.hpp"

namespace arm
{

namespace pipe
{

std::unique_ptr<ISendTimelinePacket> TimelinePacketWriterFactory::GetSendTimelinePacket() const
{
    return std::make_unique<SendTimelinePacket>(m_BufferManager);
}

} // namespace pipe

} // namespace arm
