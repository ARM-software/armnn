//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Exceptions.hpp>

#include <boost/log/trivial.hpp>

namespace armnn
{

namespace profiling
{

class Packet
{
public:
    Packet(uint32_t header, uint32_t length, const char* data)
    : m_Header(header), m_Length(length), m_Data(data)
          {
              m_PacketId = ((header >> 16) & 1023);
              m_PacketFamily = (header >> 26);

             if (length == 0)
             {
                 if (m_Data != nullptr)
                 {
                     throw armnn::Exception("Data should be null");
                 }
             }
          };

    uint32_t GetHeader() const;
    uint32_t GetPacketFamily() const;
    uint32_t GetPacketId() const;
    uint32_t GetLength() const;
    const char* GetData();

    uint32_t GetPacketClass() const;
    uint32_t GetPacketType() const;

private:
    uint32_t m_Header;
    uint32_t m_PacketFamily;
    uint32_t m_PacketId;
    uint32_t m_Length;
    const char* m_Data;
};

} // namespace profiling

} // namespace armnn
