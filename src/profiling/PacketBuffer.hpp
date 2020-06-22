//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IPacketBuffer.hpp"

#include <memory>

namespace armnn
{

namespace profiling
{

class PacketBuffer : public IPacketBuffer
{
public:
    PacketBuffer(unsigned int maxSize);

    ~PacketBuffer() {}

    const unsigned char* GetReadableData() const  override;

    unsigned int GetSize() const  override;

    void MarkRead() override;

    void Commit(unsigned int size)  override;

    void Release() override;

    unsigned char* GetWritableData() override;

    void Destroy() override;

private:
    unsigned int m_MaxSize;
    unsigned int m_Size;
    std::unique_ptr<unsigned char[]> m_Data;
};

} // namespace profiling

} // namespace armnn
