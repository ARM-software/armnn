//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Exceptions.hpp>

#include "IPacketBuffer.hpp"

#include <boost/numeric/conversion/cast.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace armnn
{

namespace profiling
{

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

template <typename SwTracePolicy>
bool IsValidSwTraceString(const std::string& s)
{
    // Check that all the characters in the given string conform to the given policy
    return std::all_of(s.begin(), s.end(), [](unsigned char c)
    {
        return SwTracePolicy::IsValidChar(c);
    });
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
    size_t s_size = s.size() + 1; // The size of the string (in chars) plus the null-terminator
    size_t uint32_t_size = sizeof(uint32_t);
    size_t outBufferSize = 1 + s_size / uint32_t_size + (s_size % uint32_t_size != 0 ? 1 : 0);
    outputBuffer.resize(outBufferSize, '\0');

    // Write the SWTrace string to the output buffer
    outputBuffer[0] = boost::numeric_cast<uint32_t>(s_size);
    std::memcpy(outputBuffer.data() + 1, s.data(), s_size);

    return true;
}

uint16_t GetNextUid(bool peekOnly = false);

std::vector<uint16_t> GetNextCounterUids(uint16_t cores);

void WriteUint64(const std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int offset, uint64_t value);

void WriteUint32(const std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int offset, uint32_t value);

void WriteUint16(const std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int offset, uint16_t value);

void WriteUint64(unsigned char* buffer, unsigned int offset, uint64_t value);

void WriteUint32(unsigned char* buffer, unsigned int offset, uint32_t value);

void WriteUint16(unsigned char* buffer, unsigned int offset, uint16_t value);

uint64_t ReadUint64(const std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int offset);

uint32_t ReadUint32(const std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int offset);

uint16_t ReadUint16(const std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int offset);

uint8_t ReadUint8(const std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int offset);

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

TimelinePacketStatus WriteTimelineLabelBinaryPacket(uint64_t profilingGuid,
                                                    const std::string& label,
                                                    unsigned char* buffer,
                                                    unsigned int bufferSize,
                                                    unsigned int& numberOfBytesWritten);

TimelinePacketStatus WriteTimelineEntityBinaryPacket(uint64_t profilingGuid,
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

class BufferExhaustion : public armnn::Exception
{
    using Exception::Exception;
};

} // namespace profiling

} // namespace armnn
