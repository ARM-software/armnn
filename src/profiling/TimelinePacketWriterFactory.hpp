//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IBufferManager.hpp"
#include "armnn/profiling/ISendTimelinePacket.hpp"

#include <memory>

namespace armnn
{

namespace profiling
{

class TimelinePacketWriterFactory
{
public:
    TimelinePacketWriterFactory(IBufferManager& bufferManager) : m_BufferManager(bufferManager) {}

    std::unique_ptr<ISendTimelinePacket> GetSendTimelinePacket() const;

private:
    IBufferManager& m_BufferManager;
};

} // namespace profiling

} // namespace armnn
