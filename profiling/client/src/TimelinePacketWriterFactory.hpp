//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IBufferManager.hpp"

#include <client/include/ISendTimelinePacket.hpp>

#include <memory>

namespace arm
{

namespace pipe
{

class TimelinePacketWriterFactory
{
public:
    TimelinePacketWriterFactory(IBufferManager& bufferManager) : m_BufferManager(bufferManager) {}

    std::unique_ptr<ISendTimelinePacket> GetSendTimelinePacket() const;

private:
    IBufferManager& m_BufferManager;
};

} // namespace pipe

} // namespace arm
