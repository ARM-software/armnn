//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IConsumer.hpp"
#include "IPacketBuffer.hpp"

#include <memory>

#define MAX_METADATA_PACKET_LENGTH 4096

namespace arm
{

namespace pipe
{

class IBufferManager
{
public:
    virtual ~IBufferManager() {}

    virtual IPacketBufferPtr Reserve(unsigned int requestedSize, unsigned int& reservedSize) = 0;

    virtual void Commit(IPacketBufferPtr& packetBuffer, unsigned int size, bool notifyConsumer = true) = 0;

    virtual void Release(IPacketBufferPtr& packetBuffer) = 0;

    virtual IPacketBufferPtr GetReadableBuffer() = 0;

    virtual void MarkRead(IPacketBufferPtr& packetBuffer) = 0;

    virtual void SetConsumer(IConsumer* consumer) = 0;

    virtual void FlushReadList() = 0;
};

} // namespace pipe

} // namespace arm
