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

void WriteUint64(const std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int offset, uint64_t value)
{
    BOOST_ASSERT(packetBuffer);

    WriteUint64(packetBuffer->GetWritableData(), offset, value);
}

void WriteUint32(const std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int offset, uint32_t value)
{
    BOOST_ASSERT(packetBuffer);

    WriteUint32(packetBuffer->GetWritableData(), offset, value);
}

void WriteUint16(const std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int offset, uint16_t value)
{
    BOOST_ASSERT(packetBuffer);

    WriteUint16(packetBuffer->GetWritableData(), offset, value);
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

uint64_t ReadUint64(const std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int offset)
{
    BOOST_ASSERT(packetBuffer);

    return ReadUint64(packetBuffer->GetReadableData(), offset);
}

uint32_t ReadUint32(const std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int offset)
{
    BOOST_ASSERT(packetBuffer);

    return ReadUint32(packetBuffer->GetReadableData(), offset);
}

uint16_t ReadUint16(const std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int offset)
{
    BOOST_ASSERT(packetBuffer);

    return ReadUint16(packetBuffer->GetReadableData(), offset);
}

uint8_t ReadUint8(const std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int offset)
{
    BOOST_ASSERT(packetBuffer);

    return ReadUint8(packetBuffer->GetReadableData(), offset);
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

uint8_t ReadUint8(const unsigned char* buffer, unsigned int offset)
{
    BOOST_ASSERT(buffer);

    return buffer[offset];
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

TimelinePacketStatus WriteTimelineLabelBinaryPacket(uint64_t profilingGuid,
                                                    const std::string& label,
                                                    unsigned char* buffer,
                                                    unsigned int bufferSize,
                                                    unsigned int& numberOfBytesWritten)
{
    // Initialize the ouput value
    numberOfBytesWritten = 0;

    // Check that the given buffer is valid
    if (buffer == nullptr || bufferSize == 0)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Convert the label into a SWTrace string
    std::vector<uint32_t> swTraceLabel;
    bool result = StringToSwTraceString<SwTraceCharPolicy>(label, swTraceLabel);
    if (!result)
    {
        return TimelinePacketStatus::Error;
    }

    // Calculate the size of the SWTrace string label (in bytes)
    unsigned int swTraceLabelSize = boost::numeric_cast<unsigned int>(swTraceLabel.size()) * uint32_t_size;

    // Calculate the length of the data (in bytes)
    unsigned int timelineLabelPacketDataLength = uint64_t_size +   // Profiling GUID
                                                 swTraceLabelSize; // Label

    // Calculate the timeline binary packet size (in bytes)
    unsigned int timelineLabelPacketSize = 2 * uint32_t_size +            // Header (2 words)
                                           timelineLabelPacketDataLength; // Profiling GUID + label

    // Check whether the timeline binary packet fits in the given buffer
    if (timelineLabelPacketSize > bufferSize)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Packet header word 0:
    // 26:31 [6] packet_family: timeline Packet Family, value 0b000001
    // 19:25 [7] packet_class: packet class
    // 16:18 [3] packet_type: packet type
    // 8:15  [8] reserved: all zeros
    // 0:7   [8] stream_id: stream identifier
    uint32_t packetFamily = 1;
    uint32_t packetClass  = 0;
    uint32_t packetType   = 1;
    uint32_t streamId     = 0;
    uint32_t packetHeaderWord0 = ((packetFamily & 0x0000003F) << 26) |
                                 ((packetClass  & 0x0000007F) << 19) |
                                 ((packetType   & 0x00000007) << 16) |
                                 ((streamId     & 0x00000007) <<  0);

    // Packet header word 1:
    // 25:31 [7]  reserved: all zeros
    // 24    [1]  sequence_numbered: when non-zero the 4 bytes following the header is a u32 sequence number
    // 0:23  [24] data_length: unsigned 24-bit integer. Length of data, in bytes. Zero is permitted
    uint32_t sequenceNumbered = 0;
    uint32_t dataLength       = boost::numeric_cast<uint32_t>(timelineLabelPacketDataLength); // Profiling GUID + label
    uint32_t packetHeaderWord1 = ((sequenceNumbered & 0x00000001) << 24) |
                                 ((dataLength       & 0x00FFFFFF) <<  0);

    // Initialize the offset for writing in the buffer
    unsigned int offset = 0;

    // Write the timeline binary packet header to the buffer
    WriteUint32(buffer, offset, packetHeaderWord0);
    offset += uint32_t_size;
    WriteUint32(buffer, offset, packetHeaderWord1);
    offset += uint32_t_size;

    // Write the timeline binary packet payload to the buffer
    WriteUint64(buffer, offset, profilingGuid); // Profiling GUID
    offset += uint64_t_size;
    for (uint32_t swTraceLabelWord : swTraceLabel)
    {
        WriteUint32(buffer, offset, swTraceLabelWord); // Label
        offset += uint32_t_size;
    }

    // Update the number of bytes written
    numberOfBytesWritten = timelineLabelPacketSize;

    return TimelinePacketStatus::Ok;
}

TimelinePacketStatus WriteTimelineEntityBinaryPacket(uint64_t profilingGuid,
                                                     unsigned char* buffer,
                                                     unsigned int bufferSize,
                                                     unsigned int& numberOfBytesWritten)
{
    // Initialize the ouput value
    numberOfBytesWritten = 0;

    // Check that the given buffer is valid
    if (buffer == nullptr || bufferSize == 0)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Calculate the length of the data (in bytes)
    unsigned int timelineEntityPacketDataLength = uint64_t_size;   // Profiling GUID


    // Calculate the timeline binary packet size (in bytes)
    unsigned int timelineEntityPacketSize = 2 * uint32_t_size +             // Header (2 words)
                                           timelineEntityPacketDataLength; // Profiling GUID

    // Check whether the timeline binary packet fits in the given buffer
    if (timelineEntityPacketSize > bufferSize)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Packet header word 0:
    // 26:31 [6] packet_family: timeline Packet Family, value 0b000001
    // 19:25 [7] packet_class: packet class
    // 16:18 [3] packet_type: packet type
    // 8:15  [8] reserved: all zeros
    // 0:7   [8] stream_id: stream identifier
    uint32_t packetFamily = 1;
    uint32_t packetClass  = 0;
    uint32_t packetType   = 1;
    uint32_t streamId     = 0;
    uint32_t packetHeaderWord0 = ((packetFamily & 0x0000003F) << 26) |
                                 ((packetClass  & 0x0000007F) << 19) |
                                 ((packetType   & 0x00000007) << 16) |
                                 ((streamId     & 0x00000007) <<  0);

    // Packet header word 1:
    // 25:31 [7]  reserved: all zeros
    // 24    [1]  sequence_numbered: when non-zero the 4 bytes following the header is a u32 sequence number
    // 0:23  [24] data_length: unsigned 24-bit integer. Length of data, in bytes. Zero is permitted
    uint32_t sequenceNumbered = 0;
    uint32_t dataLength       = boost::numeric_cast<uint32_t>(timelineEntityPacketDataLength); // Profiling GUID
    uint32_t packetHeaderWord1 = ((sequenceNumbered & 0x00000001) << 24) |
                                 ((dataLength       & 0x00FFFFFF) <<  0);

    // Initialize the offset for writing in the buffer
    unsigned int offset = 0;

    // Write the timeline binary packet header to the buffer
    WriteUint32(buffer, offset, packetHeaderWord0);
    offset += uint32_t_size;
    WriteUint32(buffer, offset, packetHeaderWord1);
    offset += uint32_t_size;

    // Write the timeline binary packet payload to the buffer
    WriteUint64(buffer, offset, profilingGuid); // Profiling GUID

    // Update the number of bytes written
    numberOfBytesWritten = timelineEntityPacketSize;

    return TimelinePacketStatus::Ok;
}

} // namespace profiling

} // namespace armnn
