//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <memory>

namespace arm
{

namespace pipe
{

class IReadOnlyPacketBuffer // interface used by the read thread
{
public:
    virtual ~IReadOnlyPacketBuffer() {}

    virtual const unsigned char* GetReadableData() const = 0;

    virtual unsigned int GetSize() const = 0;

    virtual void MarkRead() = 0;
};

class IPacketBuffer : public IReadOnlyPacketBuffer // interface used by code that writes binary packets
{
public:
    virtual ~IPacketBuffer() {}

    virtual void Commit(unsigned int size) = 0;

    virtual void Release() = 0;

    virtual unsigned char* GetWritableData() = 0;

    /// release the memory held and reset internal point to null.
    /// After this function is invoked the PacketBuffer is unusable.
    virtual void Destroy() = 0;
};

using IPacketBufferPtr = std::unique_ptr<IPacketBuffer>;

} // namespace pipe

} // namespace arm
