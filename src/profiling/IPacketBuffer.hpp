//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{

namespace profiling
{

class IReadOnlyPacketBuffer // interface used by the read thread
{
public:
    virtual ~IReadOnlyPacketBuffer() {}

    virtual const unsigned char* const GetReadableData() const = 0;

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
};

} // namespace profiling

} // namespace armnn