//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Exceptions.hpp>
#include <armnn/profiling/ISendTimelinePacket.hpp>

#include "ICounterDirectory.hpp"
#include "IPacketBuffer.hpp"

#include <boost/numeric/conversion/cast.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace armnn
{

namespace profiling
{

struct SwTraceHeader
{
    uint8_t m_StreamVersion;
    uint8_t m_PointerBytes;
    uint8_t m_ThreadIdBytes;
};

struct SwTraceMessage
{
    uint32_t m_Id;
    std::string m_Name;
    std::string m_UiName;
    std::vector<char> m_ArgTypes;
    std::vector<std::string> m_ArgNames;
};

struct SwTraceCharPolicy
{
    static bool IsValidChar(unsigned char c)
    {
        // Check that the given character has ASCII 7-bit encoding
        return c < 128;
    }
};

struct SwTraceNameCharPolicy
{
    static bool IsValidChar(unsigned char c)
    {
        // Check that the given character has ASCII 7-bit encoding, alpha-numeric and underscore only
        return c < 128 && (std::isalnum(c) || c == '_');
    }
};

struct SwTraceTypeCharPolicy
{
    static bool IsValidChar(unsigned char c)
    {
        // Check that the given character is among the allowed ones
        switch (c)
        {
        case '@':
        case 't':
        case 'i':
        case 'I':
        case 'l':
        case 'L':
        case 'F':
        case 'p':
        case 's':
            return true; // Valid char
        default:
            return false; // Invalid char
        }
    }
};

template <typename SwTracePolicy>
bool IsValidSwTraceString(const std::string& s)
{
    // Check that all the characters in the given string conform to the given policy
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return SwTracePolicy::IsValidChar(c); });
}

template <typename SwTracePolicy>
bool StringToSwTraceString(const std::string& s, std::vector<uint32_t>& outputBuffer)
{
    // Converts the given string to an SWTrace "string" (i.e. a string of "chars"), and writes it into
    // the given buffer including the null-terminator. It also pads it to the next uint32_t if necessary

    // Clear the output buffer
    outputBuffer.clear();

    // Check that the given string is a valid SWTrace "string" (i.e. a string of "chars")
    if (!IsValidSwTraceString<SwTracePolicy>(s))
    {
        return false;
    }

    // Prepare the output buffer
    size_t s_size        = s.size() + 1;    // The size of the string (in chars) plus the null-terminator
    size_t uint32_t_size = sizeof(uint32_t);
    size_t outBufferSize = 1 + s_size / uint32_t_size + (s_size % uint32_t_size != 0 ? 1 : 0);
    outputBuffer.resize(outBufferSize, '\0');

    // Write the SWTrace string to the output buffer
    outputBuffer[0] = boost::numeric_cast<uint32_t>(s_size);
    std::memcpy(outputBuffer.data() + 1, s.data(), s_size);

    return true;
}

template <typename SwTracePolicy,
          typename SwTraceBuffer = std::vector<uint32_t>>
bool ConvertDirectoryComponent(const std::string& directoryComponent, SwTraceBuffer& swTraceBuffer)
{
    // Convert the directory component using the given policy
    SwTraceBuffer tempSwTraceBuffer;
    bool result = StringToSwTraceString<SwTracePolicy>(directoryComponent, tempSwTraceBuffer);
    if (!result)
    {
        return false;
    }

    swTraceBuffer.insert(swTraceBuffer.end(), tempSwTraceBuffer.begin(), tempSwTraceBuffer.end());

    return true;
}

uint16_t GetNextUid(bool peekOnly = false);

std::vector<uint16_t> GetNextCounterUids(uint16_t firstUid, uint16_t cores);

void WriteBytes(const IPacketBuffer& packetBuffer, unsigned int offset, const void* value, unsigned int valueSize);

uint32_t ConstructHeader(uint32_t packetFamily, uint32_t packetId);

uint32_t ConstructHeader(uint32_t packetFamily, uint32_t packetClass, uint32_t packetType);

void WriteUint64(const IPacketBufferPtr& packetBuffer, unsigned int offset, uint64_t value);

void WriteUint32(const IPacketBufferPtr& packetBuffer, unsigned int offset, uint32_t value);

void WriteUint16(const IPacketBufferPtr& packetBuffer, unsigned int offset, uint16_t value);

void WriteUint8(const IPacketBufferPtr& packetBuffer, unsigned int offset, uint8_t value);

void WriteBytes(unsigned char* buffer, unsigned int offset, const void* value, unsigned int valueSize);

void WriteUint64(unsigned char* buffer, unsigned int offset, uint64_t value);

void WriteUint32(unsigned char* buffer, unsigned int offset, uint32_t value);

void WriteUint16(unsigned char* buffer, unsigned int offset, uint16_t value);

void WriteUint8(unsigned char* buffer, unsigned int offset, uint8_t value);

void ReadBytes(const IPacketBufferPtr& packetBuffer, unsigned int offset, unsigned int valueSize, uint8_t outValue[]);

uint64_t ReadUint64(const IPacketBufferPtr& packetBuffer, unsigned int offset);

uint32_t ReadUint32(const IPacketBufferPtr& packetBuffer, unsigned int offset);

uint16_t ReadUint16(const IPacketBufferPtr& packetBuffer, unsigned int offset);

uint8_t ReadUint8(const IPacketBufferPtr& packetBuffer, unsigned int offset);

void ReadBytes(const unsigned char* buffer, unsigned int offset, unsigned int valueSize, uint8_t outValue[]);

uint64_t ReadUint64(unsigned const char* buffer, unsigned int offset);

uint32_t ReadUint32(unsigned const char* buffer, unsigned int offset);

uint16_t ReadUint16(unsigned const char* buffer, unsigned int offset);

uint8_t ReadUint8(unsigned const char* buffer, unsigned int offset);

std::string GetSoftwareInfo();

std::string GetSoftwareVersion();

std::string GetHardwareVersion();

std::string GetProcessName();

enum class TimelinePacketStatus
{
    Ok,
    Error,
    BufferExhaustion
};

uint32_t CalculateSizeOfPaddedSwString(const std::string& str);

SwTraceMessage ReadSwTraceMessage(const unsigned char*, unsigned int& offset);

TimelinePacketStatus WriteTimelineLabelBinaryPacket(uint64_t profilingGuid,
                                                    const std::string& label,
                                                    unsigned char* buffer,
                                                    unsigned int bufferSize,
                                                    unsigned int& numberOfBytesWritten);

TimelinePacketStatus WriteTimelineEntityBinaryPacket(uint64_t profilingGuid,
                                                     unsigned char* buffer,
                                                     unsigned int bufferSize,
                                                     unsigned int& numberOfBytesWritten);

TimelinePacketStatus WriteTimelineRelationshipBinaryPacket(ProfilingRelationshipType relationshipType,
                                                           uint64_t relationshipGuid,
                                                           uint64_t headGuid,
                                                           uint64_t tailGuid,
                                                           unsigned char* buffer,
                                                           unsigned int bufferSize,
                                                           unsigned int& numberOfBytesWritten);

TimelinePacketStatus WriteTimelineMessageDirectoryPackage(unsigned char* buffer,
                                                          unsigned int bufferSize,
                                                          unsigned int& numberOfBytesWritten);

TimelinePacketStatus WriteTimelineEventClassBinaryPacket(uint64_t profilingGuid,
                                                         unsigned char* buffer,
                                                         unsigned int bufferSize,
                                                         unsigned int& numberOfBytesWritten);

TimelinePacketStatus WriteTimelineEventBinaryPacket(uint64_t timestamp,
                                                    std::thread::id threadId,
                                                    uint64_t profilingGuid,
                                                    unsigned char* buffer,
                                                    unsigned int bufferSize,
                                                    unsigned int& numberOfBytesWritten);

std::string CentreAlignFormatting(const std::string& stringToPass, const int spacingWidth);

void PrintCounterDirectory(ICounterDirectory& counterDirectory);

class BufferExhaustion : public armnn::Exception
{
    using Exception::Exception;
};

uint64_t GetTimestamp();

} // namespace profiling

} // namespace armnn

namespace std
{

bool operator==(const std::vector<uint8_t>& left, std::thread::id right);

} // namespace std
