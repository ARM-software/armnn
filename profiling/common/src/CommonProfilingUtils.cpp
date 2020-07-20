//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <common/include/Assert.hpp>
#include <common/include/CommonProfilingUtils.hpp>

#include <sstream>

namespace arm
{

namespace pipe
{
void ReadBytes(const unsigned char* buffer, unsigned int offset, unsigned int valueSize, uint8_t outValue[])
{
    ARM_PIPE_ASSERT(buffer);
    ARM_PIPE_ASSERT(outValue);

    for (unsigned int i = 0; i < valueSize; i++, offset++)
    {
        outValue[i] = static_cast<uint8_t>(buffer[offset]);
    }
}

uint64_t ReadUint64(const unsigned char* buffer, unsigned int offset)
{
    ARM_PIPE_ASSERT(buffer);

    uint64_t value = 0;
    value  = static_cast<uint64_t>(buffer[offset]);
    value |= static_cast<uint64_t>(buffer[offset + 1]) << 8;
    value |= static_cast<uint64_t>(buffer[offset + 2]) << 16;
    value |= static_cast<uint64_t>(buffer[offset + 3]) << 24;
    value |= static_cast<uint64_t>(buffer[offset + 4]) << 32;
    value |= static_cast<uint64_t>(buffer[offset + 5]) << 40;
    value |= static_cast<uint64_t>(buffer[offset + 6]) << 48;
    value |= static_cast<uint64_t>(buffer[offset + 7]) << 56;

    return value;
}

uint32_t ReadUint32(const unsigned char* buffer, unsigned int offset)
{
    ARM_PIPE_ASSERT(buffer);

    uint32_t value = 0;
    value  = static_cast<uint32_t>(buffer[offset]);
    value |= static_cast<uint32_t>(buffer[offset + 1]) << 8;
    value |= static_cast<uint32_t>(buffer[offset + 2]) << 16;
    value |= static_cast<uint32_t>(buffer[offset + 3]) << 24;
    return value;
}

uint16_t ReadUint16(const unsigned char* buffer, unsigned int offset)
{
    ARM_PIPE_ASSERT(buffer);

    uint32_t value = 0;
    value  = static_cast<uint32_t>(buffer[offset]);
    value |= static_cast<uint32_t>(buffer[offset + 1]) << 8;
    return static_cast<uint16_t>(value);
}

uint8_t ReadUint8(const unsigned char* buffer, unsigned int offset)
{
    ARM_PIPE_ASSERT(buffer);

    return buffer[offset];
}

void WriteBytes(unsigned char* buffer, unsigned int offset, const void* value, unsigned int valueSize)
{
    ARM_PIPE_ASSERT(buffer);
    ARM_PIPE_ASSERT(value);

    for (unsigned int i = 0; i < valueSize; i++, offset++)
    {
        buffer[offset] = *(reinterpret_cast<const unsigned char*>(value) + i);
    }
}

void WriteUint64(unsigned char* buffer, unsigned int offset, uint64_t value)
{
    ARM_PIPE_ASSERT(buffer);

    buffer[offset]     = static_cast<unsigned char>(value & 0xFF);
    buffer[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
    buffer[offset + 2] = static_cast<unsigned char>((value >> 16) & 0xFF);
    buffer[offset + 3] = static_cast<unsigned char>((value >> 24) & 0xFF);
    buffer[offset + 4] = static_cast<unsigned char>((value >> 32) & 0xFF);
    buffer[offset + 5] = static_cast<unsigned char>((value >> 40) & 0xFF);
    buffer[offset + 6] = static_cast<unsigned char>((value >> 48) & 0xFF);
    buffer[offset + 7] = static_cast<unsigned char>((value >> 56) & 0xFF);
}

void WriteUint32(unsigned char* buffer, unsigned int offset, uint32_t value)
{
    ARM_PIPE_ASSERT(buffer);

    buffer[offset]     = static_cast<unsigned char>(value & 0xFF);
    buffer[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
    buffer[offset + 2] = static_cast<unsigned char>((value >> 16) & 0xFF);
    buffer[offset + 3] = static_cast<unsigned char>((value >> 24) & 0xFF);
}

void WriteUint16(unsigned char* buffer, unsigned int offset, uint16_t value)
{
    ARM_PIPE_ASSERT(buffer);

    buffer[offset]     = static_cast<unsigned char>(value & 0xFF);
    buffer[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
}

void WriteUint8(unsigned char* buffer, unsigned int offset, uint8_t value)
{
    ARM_PIPE_ASSERT(buffer);

    buffer[offset] = static_cast<unsigned char>(value);
}

std::string CentreAlignFormatting(const std::string& stringToPass, const int spacingWidth)
{
    std::stringstream outputStream, centrePadding;
    int padding = spacingWidth - static_cast<int>(stringToPass.size());

    for (int i = 0; i < padding / 2; ++i)
    {
        centrePadding << " ";
    }

    outputStream << centrePadding.str() << stringToPass << centrePadding.str();

    if (padding > 0 && padding %2 != 0)
    {
        outputStream << " ";
    }

    return outputStream.str();
}


} // namespace pipe
} // namespace arm