//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Exceptions.hpp>
#include <armnn/profiling/ISendTimelinePacket.hpp>

#include <armnn/utility/NumericCast.hpp>

#include "ICounterDirectory.hpp"
#include "IPacketBuffer.hpp"

#include <common/include/Packet.hpp>

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

constexpr unsigned int ThreadIdSize = sizeof(int); // Is platform dependent

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

std::pair<uint32_t, uint32_t> CreateTimelinePacketHeader(uint32_t packetFamily,
                                                         uint32_t packetClass,
                                                         uint32_t packetType,
                                                         uint32_t streamId,
                                                         uint32_t sequenceNumbered,
                                                         uint32_t dataLength);

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

TimelinePacketStatus WriteTimelineEntityBinary(uint64_t profilingGuid,
                                               unsigned char* buffer,
                                               unsigned int bufferSize,
                                               unsigned int& numberOfBytesWritten);

TimelinePacketStatus WriteTimelineRelationshipBinary(ProfilingRelationshipType relationshipType,
                                                     uint64_t relationshipGuid,
                                                     uint64_t headGuid,
                                                     uint64_t tailGuid,
                                                     uint64_t attributeGuid,
                                                     unsigned char* buffer,
                                                     unsigned int bufferSize,
                                                     unsigned int& numberOfBytesWritten);

TimelinePacketStatus WriteTimelineMessageDirectoryPackage(unsigned char* buffer,
                                                          unsigned int bufferSize,
                                                          unsigned int& numberOfBytesWritten);

TimelinePacketStatus WriteTimelineEventClassBinary(uint64_t profilingGuid,
                                                   uint64_t nameGuid,
                                                   unsigned char* buffer,
                                                   unsigned int bufferSize,
                                                   unsigned int& numberOfBytesWritten);

TimelinePacketStatus WriteTimelineEventBinary(uint64_t timestamp,
                                              int threadId,
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

arm::pipe::Packet ReceivePacket(const unsigned char* buffer, uint32_t length);

} // namespace profiling

} // namespace armnn

namespace std
{

bool operator==(const std::vector<uint8_t>& left, int right);

} // namespace std
