//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingUtils.hpp"

#include <boost/assert.hpp>

namespace armnn
{

namespace profiling
{

void WriteUint32(unsigned char* buffer, unsigned int offset, uint32_t value)
{
    BOOST_ASSERT(buffer);

    buffer[offset] = static_cast<unsigned char>(value & 0xFF);
    buffer[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
    buffer[offset + 2] = static_cast<unsigned char>((value >> 16) & 0xFF);
    buffer[offset + 3] = static_cast<unsigned char>((value >> 24) & 0xFF);
}

void WriteUint16(unsigned char* buffer, unsigned int offset, uint16_t value)
{
    BOOST_ASSERT(buffer != nullptr);

    buffer[offset] = static_cast<unsigned char>(value & 0xFF);
    buffer[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
}

uint32_t ReadUint32(const unsigned char* buffer, unsigned int offset)
{
    BOOST_ASSERT(buffer);

    uint32_t value = 0;
    value = static_cast<uint32_t>(buffer[offset]);
    value |= static_cast<uint32_t>(buffer[offset + 1]) << 8;
    value |= static_cast<uint32_t>(buffer[offset + 2]) << 16;
    value |= static_cast<uint32_t>(buffer[offset + 3]) << 24;
    return value;
}

uint16_t ReadUint16(const unsigned char* buffer, unsigned int offset)
{
    BOOST_ASSERT(buffer);

    uint32_t value = 0;
    value = static_cast<uint32_t>(buffer[offset]);
    value |= static_cast<uint32_t>(buffer[offset + 1]) << 8;
    return static_cast<uint16_t>(value);
}

} // namespace profiling

} // namespace armnn