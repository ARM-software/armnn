//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PacketBuffer.hpp"

#include <common/include/ProfilingException.hpp>

namespace arm
{

namespace pipe
{

PacketBuffer::PacketBuffer(unsigned int maxSize)
    : m_MaxSize(maxSize)
    , m_Size(0)
{
    m_Data = std::make_unique<unsigned char[]>(m_MaxSize);
}

const unsigned char* PacketBuffer::GetReadableData() const
{
    return m_Data.get();
}

unsigned int PacketBuffer::GetSize() const
{
    return m_Size;
}

void PacketBuffer::MarkRead()
{
    m_Size = 0;
}

void PacketBuffer::Commit(unsigned int size)
{
    if (size > m_MaxSize)
    {
        throw arm::pipe::ProfilingException("Cannot commit [" + std::to_string(size) +
            "] bytes which is more than the maximum size of the buffer [" + std::to_string(m_MaxSize) + "]");
    }
    m_Size = size;
}

void PacketBuffer::Release()
{
    m_Size = 0;
}

unsigned char* PacketBuffer::GetWritableData()
{
    return m_Data.get();
}

void PacketBuffer::Destroy()
{
    m_Data.reset(nullptr);
    m_Size = 0;
    m_MaxSize = 0;
}

} // namespace pipe

} // namespace arm
