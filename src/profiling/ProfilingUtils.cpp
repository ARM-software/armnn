//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingUtils.hpp"

#include <armnn/Version.hpp>
#include <armnn/Conversion.hpp>

#include <boost/assert.hpp>

#include <fstream>
#include <limits>

namespace armnn
{

namespace profiling
{

namespace
{

void ThrowIfCantGenerateNextUid(uint16_t uid, uint16_t cores = 0)
{
    // Check that it is possible to generate the next UID without causing an overflow
    switch (cores)
    {
    case 0:
    case 1:
        // Number of cores not specified or set to 1 (a value of zero indicates the device is not capable of
        // running multiple parallel workloads and will not provide multiple streams of data for each event)
        if (uid == std::numeric_limits<uint16_t>::max())
        {
            throw RuntimeException("Generating the next UID for profiling would result in an overflow");
        }
        break;
    default: // cores > 1
        // Multiple cores available, as max_counter_uid has to be set to: counter_uid + cores - 1, the maximum
        // allowed value for a counter UID is consequently: uint16_t_max - cores + 1
        if (uid >= std::numeric_limits<uint16_t>::max() - cores + 1)
        {
            throw RuntimeException("Generating the next UID for profiling would result in an overflow");
        }
        break;
    }
}

} // Anonymous namespace

uint16_t GetNextUid(bool peekOnly)
{
    // The UID used for profiling objects and events. The first valid UID is 1, as 0 is a reserved value
    static uint16_t uid = 1;

    // Check that it is possible to generate the next UID without causing an overflow (throws in case of error)
    ThrowIfCantGenerateNextUid(uid);

    if (peekOnly)
    {
        // Peek only
        return uid;
    }
    else
    {
        // Get the next UID
        return uid++;
    }
}

std::vector<uint16_t> GetNextCounterUids(uint16_t cores)
{
    // The UID used for counters only. The first valid UID is 0
    static uint16_t counterUid = 0;

    // Check that it is possible to generate the next counter UID without causing an overflow (throws in case of error)
    ThrowIfCantGenerateNextUid(counterUid, cores);

    // Get the next counter UIDs
    size_t counterUidsSize = cores == 0 ? 1 : cores;
    std::vector<uint16_t> counterUids(counterUidsSize, 0);
    for (size_t i = 0; i < counterUidsSize; i++)
    {
        counterUids[i] = counterUid++;
    }
    return counterUids;
}

void WriteUint64(unsigned char* buffer, unsigned int offset, uint64_t value)
{
    BOOST_ASSERT(buffer);

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
    BOOST_ASSERT(buffer);

    buffer[offset]     = static_cast<unsigned char>(value & 0xFF);
    buffer[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
    buffer[offset + 2] = static_cast<unsigned char>((value >> 16) & 0xFF);
    buffer[offset + 3] = static_cast<unsigned char>((value >> 24) & 0xFF);
}

void WriteUint16(unsigned char* buffer, unsigned int offset, uint16_t value)
{
    BOOST_ASSERT(buffer);

    buffer[offset]     = static_cast<unsigned char>(value & 0xFF);
    buffer[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
}

uint64_t ReadUint64(const unsigned char* buffer, unsigned int offset)
{
    BOOST_ASSERT(buffer);

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
    BOOST_ASSERT(buffer);

    uint32_t value = 0;
    value  = static_cast<uint32_t>(buffer[offset]);
    value |= static_cast<uint32_t>(buffer[offset + 1]) << 8;
    value |= static_cast<uint32_t>(buffer[offset + 2]) << 16;
    value |= static_cast<uint32_t>(buffer[offset + 3]) << 24;
    return value;
}

uint16_t ReadUint16(const unsigned char* buffer, unsigned int offset)
{
    BOOST_ASSERT(buffer);

    uint32_t value = 0;
    value  = static_cast<uint32_t>(buffer[offset]);
    value |= static_cast<uint32_t>(buffer[offset + 1]) << 8;
    return static_cast<uint16_t>(value);
}

std::string GetSoftwareInfo()
{
    return std::string("ArmNN");
}

std::string GetHardwareVersion()
{
    return std::string();
}

std::string GetSoftwareVersion()
{
    std::string armnnVersion(ARMNN_VERSION);
    std::string result = "Armnn " + armnnVersion.substr(2,2) + "." + armnnVersion.substr(4,2);
    return result;
}

std::string GetProcessName()
{
    std::ifstream comm("/proc/self/comm");
    std::string name;
    getline(comm, name);
    return name;
}

} // namespace profiling

} // namespace armnn
