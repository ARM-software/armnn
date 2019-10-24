//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IPacketBuffer.hpp"

#include <memory>

#define MAX_METADATA_PACKET_LENGTH 4096

namespace armnn
{

namespace profiling
{

class IBufferManager
{
public:
    virtual ~IBufferManager() {}

    virtual std::unique_ptr<IPacketBuffer> Reserve(unsigned int requestedSize, unsigned int& reservedSize) = 0;

    virtual void Commit(std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int size) = 0;

    virtual void Release(std::unique_ptr<IPacketBuffer>& packetBuffer) = 0;

    virtual std::unique_ptr<IPacketBuffer> GetReadableBuffer() = 0;

    virtual void MarkRead(std::unique_ptr<IPacketBuffer>& packetBuffer) = 0;
};

} // namespace profiling

} // namespace armnn
