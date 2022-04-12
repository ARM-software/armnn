//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingUtils.hpp"

#include <common/include/Assert.hpp>
#include <common/include/CommonProfilingUtils.hpp>
#include <common/include/NumericCast.hpp>
#include <common/include/ProfilingException.hpp>
#include <common/include/SwTrace.hpp>

#include <armnn/Version.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

namespace arm
{

namespace pipe
{

void WriteBytes(const IPacketBufferPtr& packetBuffer, unsigned int offset,  const void* value, unsigned int valueSize)
{
    ARM_PIPE_ASSERT(packetBuffer);

    WriteBytes(packetBuffer->GetWritableData(), offset, value, valueSize);
}

uint32_t ConstructHeader(uint32_t packetFamily,
                         uint32_t packetId)
{
    return (( packetFamily & 0x0000003F ) << 26 )|
           (( packetId     & 0x000003FF ) << 16 );
}

uint32_t ConstructHeader(uint32_t packetFamily, uint32_t packetClass, uint32_t packetType)
{
    return ((packetFamily & 0x0000003F) << 26) |
           ((packetClass  & 0x0000007F) << 19) |
           ((packetType   & 0x00000007) << 16);
}

void WriteUint64(const std::unique_ptr<IPacketBuffer>& packetBuffer, unsigned int offset, uint64_t value)
{
    ARM_PIPE_ASSERT(packetBuffer);

    WriteUint64(packetBuffer->GetWritableData(), offset, value);
}

void WriteUint32(const IPacketBufferPtr& packetBuffer, unsigned int offset, uint32_t value)
{
    ARM_PIPE_ASSERT(packetBuffer);

    WriteUint32(packetBuffer->GetWritableData(), offset, value);
}

void WriteUint16(const IPacketBufferPtr& packetBuffer, unsigned int offset, uint16_t value)
{
    ARM_PIPE_ASSERT(packetBuffer);

    WriteUint16(packetBuffer->GetWritableData(), offset, value);
}

void WriteUint8(const IPacketBufferPtr& packetBuffer, unsigned int offset, uint8_t value)
{
    ARM_PIPE_ASSERT(packetBuffer);

    WriteUint8(packetBuffer->GetWritableData(), offset, value);
}

void ReadBytes(const IPacketBufferPtr& packetBuffer, unsigned int offset, unsigned int valueSize, uint8_t outValue[])
{
    ARM_PIPE_ASSERT(packetBuffer);

    ReadBytes(packetBuffer->GetReadableData(), offset, valueSize, outValue);
}

uint64_t ReadUint64(const IPacketBufferPtr& packetBuffer, unsigned int offset)
{
    ARM_PIPE_ASSERT(packetBuffer);

    return ReadUint64(packetBuffer->GetReadableData(), offset);
}

uint32_t ReadUint32(const IPacketBufferPtr& packetBuffer, unsigned int offset)
{
    ARM_PIPE_ASSERT(packetBuffer);

    return ReadUint32(packetBuffer->GetReadableData(), offset);
}

uint16_t ReadUint16(const IPacketBufferPtr& packetBuffer, unsigned int offset)
{
    ARM_PIPE_ASSERT(packetBuffer);

    return ReadUint16(packetBuffer->GetReadableData(), offset);
}

uint8_t ReadUint8(const IPacketBufferPtr& packetBuffer, unsigned int offset)
{
    ARM_PIPE_ASSERT(packetBuffer);

    return ReadUint8(packetBuffer->GetReadableData(), offset);
}

std::string GetProcessName()
{
    std::ifstream comm("/proc/self/comm");
    std::string name;
    getline(comm, name);
    return name;
}

/// Creates a timeline packet header
///
/// \params
///   packetFamiliy     Timeline Packet Family
///   packetClass       Timeline Packet Class
///   packetType        Timeline Packet Type
///   streamId          Stream identifier
///   seqeunceNumbered  When non-zero the 4 bytes following the header is a u32 sequence number
///   dataLength        Unsigned 24-bit integer. Length of data, in bytes. Zero is permitted
///
/// \returns
///   Pair of uint32_t containing word0 and word1 of the header
std::pair<uint32_t, uint32_t> CreateTimelinePacketHeader(uint32_t packetFamily,
                                                         uint32_t packetClass,
                                                         uint32_t packetType,
                                                         uint32_t streamId,
                                                         uint32_t sequenceNumbered,
                                                         uint32_t dataLength)
{
    // Packet header word 0:
    // 26:31 [6] packet_family: timeline Packet Family, value 0b000001
    // 19:25 [7] packet_class: packet class
    // 16:18 [3] packet_type: packet type
    // 8:15  [8] reserved: all zeros
    // 0:7   [8] stream_id: stream identifier
    uint32_t packetHeaderWord0 = ((packetFamily & 0x0000003F) << 26) |
                                 ((packetClass  & 0x0000007F) << 19) |
                                 ((packetType   & 0x00000007) << 16) |
                                 ((streamId     & 0x00000007) <<  0);

    // Packet header word 1:
    // 25:31 [7]  reserved: all zeros
    // 24    [1]  sequence_numbered: when non-zero the 4 bytes following the header is a u32 sequence number
    // 0:23  [24] data_length: unsigned 24-bit integer. Length of data, in bytes. Zero is permitted
    uint32_t packetHeaderWord1 = ((sequenceNumbered & 0x00000001) << 24) |
                                 ((dataLength       & 0x00FFFFFF) <<  0);

    return std::make_pair(packetHeaderWord0, packetHeaderWord1);
}

/// Creates a packet header for the timeline messages:
/// * declareLabel
/// * declareEntity
/// * declareEventClass
/// * declareRelationship
/// * declareEvent
///
/// \param
///   dataLength The length of the message body in bytes
///
/// \returns
///   Pair of uint32_t containing word0 and word1 of the header
std::pair<uint32_t, uint32_t> CreateTimelineMessagePacketHeader(unsigned int dataLength)
{
    return CreateTimelinePacketHeader(1,           // Packet family
                                      0,           // Packet class
                                      1,           // Packet type
                                      0,           // Stream id
                                      0,           // Sequence number
                                      dataLength); // Data length
}

TimelinePacketStatus WriteTimelineLabelBinaryPacket(uint64_t profilingGuid,
                                                    const std::string& label,
                                                    unsigned char* buffer,
                                                    unsigned int remainingBufferSize,
                                                    unsigned int& numberOfBytesWritten)
{
    // Initialize the output value
    numberOfBytesWritten = 0;

    // Check that the given buffer is valid
    if (buffer == nullptr || remainingBufferSize == 0)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Convert the label into a SWTrace string
    std::vector<uint32_t> swTraceLabel;
    bool result = arm::pipe::StringToSwTraceString<arm::pipe::SwTraceCharPolicy>(label, swTraceLabel);
    if (!result)
    {
        return TimelinePacketStatus::Error;
    }

    // Calculate the size of the SWTrace string label (in bytes)
    unsigned int swTraceLabelSize = arm::pipe::numeric_cast<unsigned int>(swTraceLabel.size()) * uint32_t_size;

    // Calculate the length of the data (in bytes)
    unsigned int timelineLabelPacketDataLength = uint32_t_size +   // decl_Id
                                                 uint64_t_size +   // Profiling GUID
                                                 swTraceLabelSize; // Label

    // Check whether the timeline binary packet fits in the given buffer
    if (timelineLabelPacketDataLength > remainingBufferSize)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Initialize the offset for writing in the buffer
    unsigned int offset = 0;

    // Write decl_Id to the buffer
    WriteUint32(buffer, offset, 0u);
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
    numberOfBytesWritten = timelineLabelPacketDataLength;

    return TimelinePacketStatus::Ok;
}

TimelinePacketStatus WriteTimelineEntityBinary(uint64_t profilingGuid,
                                               unsigned char* buffer,
                                               unsigned int remainingBufferSize,
                                               unsigned int& numberOfBytesWritten)
{
    // Initialize the output value
    numberOfBytesWritten = 0;

    // Check that the given buffer is valid
    if (buffer == nullptr || remainingBufferSize == 0)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Calculate the length of the data (in bytes)
    unsigned int timelineEntityDataLength = uint32_t_size + uint64_t_size;  // decl_id + Profiling GUID

    // Check whether the timeline binary packet fits in the given buffer
    if (timelineEntityDataLength > remainingBufferSize)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Initialize the offset for writing in the buffer
    unsigned int offset = 0;

    // Write the decl_Id to the buffer
    WriteUint32(buffer, offset, 1u);
    offset += uint32_t_size;

    // Write the timeline binary packet payload to the buffer
    WriteUint64(buffer, offset, profilingGuid); // Profiling GUID

    // Update the number of bytes written
    numberOfBytesWritten = timelineEntityDataLength;

    return TimelinePacketStatus::Ok;
}

TimelinePacketStatus WriteTimelineRelationshipBinary(ProfilingRelationshipType relationshipType,
                                                     uint64_t relationshipGuid,
                                                     uint64_t headGuid,
                                                     uint64_t tailGuid,
                                                     uint64_t attributeGuid,
                                                     unsigned char* buffer,
                                                     unsigned int remainingBufferSize,
                                                     unsigned int& numberOfBytesWritten)
{
    // Initialize the output value
    numberOfBytesWritten = 0;

    // Check that the given buffer is valid
    if (buffer == nullptr || remainingBufferSize == 0)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Calculate the length of the data (in bytes)
    unsigned int timelineRelationshipDataLength = uint32_t_size * 2 + // decl_id + Relationship Type
                                                  uint64_t_size * 4;  // Relationship GUID + Head GUID +
                                                                      // tail GUID + attributeGuid

    // Check whether the timeline binary fits in the given buffer
    if (timelineRelationshipDataLength > remainingBufferSize)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Initialize the offset for writing in the buffer
    unsigned int offset = 0;

    uint32_t relationshipTypeUint = 0;

    switch (relationshipType)
    {
        case ProfilingRelationshipType::RetentionLink:
            relationshipTypeUint = 0;
            break;
        case ProfilingRelationshipType::ExecutionLink:
            relationshipTypeUint = 1;
            break;
        case ProfilingRelationshipType::DataLink:
            relationshipTypeUint = 2;
            break;
        case ProfilingRelationshipType::LabelLink:
            relationshipTypeUint = 3;
            break;
        default:
            throw arm::pipe::InvalidArgumentException("Unknown relationship type given.");
    }

    // Write the timeline binary payload to the buffer
    // decl_id of the timeline message
    uint32_t declId = 3;
    WriteUint32(buffer, offset, declId); // decl_id
    offset += uint32_t_size;
    WriteUint32(buffer, offset, relationshipTypeUint); // Relationship Type
    offset += uint32_t_size;
    WriteUint64(buffer, offset, relationshipGuid); // GUID of this relationship
    offset += uint64_t_size;
    WriteUint64(buffer, offset, headGuid); // head of relationship GUID
    offset += uint64_t_size;
    WriteUint64(buffer, offset, tailGuid); // tail of relationship GUID
    offset += uint64_t_size;
    WriteUint64(buffer, offset, attributeGuid); // attribute of relationship GUID


    // Update the number of bytes written
    numberOfBytesWritten = timelineRelationshipDataLength;

    return TimelinePacketStatus::Ok;
}

TimelinePacketStatus WriteTimelineMessageDirectoryPackage(unsigned char* buffer,
                                                          unsigned int remainingBufferSize,
                                                          unsigned int& numberOfBytesWritten)
{
    // Initialize the output value
    numberOfBytesWritten = 0;

    // Check that the given buffer is valid
    if (buffer == nullptr || remainingBufferSize == 0)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Utils
    unsigned int uint8_t_size  = sizeof(uint8_t);
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // The payload/data of the packet consists of swtrace event definitions encoded according
    // to the swtrace directory specification. The messages being the five defined below:
    //
    // |  decl_id  |     decl_name       |      ui_name          |  arg_types  |            arg_names                |
    // |-----------|---------------------|-----------------------|-------------|-------------------------------------|
    // |    0      |   declareLabel      |   declare label       |    ps       |  guid,value                         |
    // |    1      |   declareEntity     |   declare entity      |    p        |  guid                               |
    // |    2      | declareEventClass   |  declare event class  |    pp       |  guid,nameGuid                      |
    // |    3      | declareRelationship | declare relationship  |    Ipppp    |  relationshipType,relationshipGuid, |
    // |           |                     |                       |             |  headGuid,tailGuid,attributeGuid    |
    // |    4      |   declareEvent      |   declare event       |    @tp      |  timestamp,threadId,eventGuid       |
    std::vector<std::vector<std::string>> timelineDirectoryMessages
    {
        { "0", "declareLabel", "declare label", "ps", "guid,value" },
        { "1", "declareEntity", "declare entity", "p", "guid" },
        { "2", "declareEventClass", "declare event class", "pp", "guid,nameGuid" },
        { "3", "declareRelationship", "declare relationship", "Ipppp",
          "relationshipType,relationshipGuid,headGuid,tailGuid,attributeGuid" },
        { "4", "declareEvent", "declare event", "@tp", "timestamp,threadId,eventGuid" }
    };

    // Build the message declarations
    std::vector<uint32_t> swTraceBuffer;
    for (const auto& directoryComponent : timelineDirectoryMessages)
    {
        // decl_id
        uint32_t declId = 0;
        try
        {
            declId = arm::pipe::numeric_cast<uint32_t>(std::stoul(directoryComponent[0]));
        }
        catch (const std::exception&)
        {
            return TimelinePacketStatus::Error;
        }
        swTraceBuffer.push_back(declId);

        bool result = true;
        result &= arm::pipe::ConvertDirectoryComponent<arm::pipe::SwTraceNameCharPolicy>(
                      directoryComponent[1], swTraceBuffer); // decl_name
        result &= arm::pipe::ConvertDirectoryComponent<arm::pipe::SwTraceCharPolicy>    (
                      directoryComponent[2], swTraceBuffer); // ui_name
        result &= arm::pipe::ConvertDirectoryComponent<arm::pipe::SwTraceTypeCharPolicy>(
                      directoryComponent[3], swTraceBuffer); // arg_types
        result &= arm::pipe::ConvertDirectoryComponent<arm::pipe::SwTraceCharPolicy>    (
                      directoryComponent[4], swTraceBuffer); // arg_names
        if (!result)
        {
            return TimelinePacketStatus::Error;
        }
    }

    unsigned int dataLength = 3 * uint8_t_size +  // Stream header (3 bytes)
                              arm::pipe::numeric_cast<unsigned int>(swTraceBuffer.size()) *
                                  uint32_t_size; // Trace directory (5 messages)

    // Calculate the timeline directory binary packet size (in bytes)
    unsigned int timelineDirectoryPacketSize = 2 * uint32_t_size + // Header (2 words)
                                               dataLength;         // Payload

    // Check whether the timeline directory binary packet fits in the given buffer
    if (timelineDirectoryPacketSize > remainingBufferSize)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Create packet header
    auto packetHeader = CreateTimelinePacketHeader(1, 0, 0, 0, 0, arm::pipe::numeric_cast<uint32_t>(dataLength));

    // Initialize the offset for writing in the buffer
    unsigned int offset = 0;

    // Write the timeline binary packet header to the buffer
    WriteUint32(buffer, offset, packetHeader.first);
    offset += uint32_t_size;
    WriteUint32(buffer, offset, packetHeader.second);
    offset += uint32_t_size;

    // Write the stream header
    uint8_t streamVersion = 4;
    uint8_t pointerBytes  = arm::pipe::numeric_cast<uint8_t>(uint64_t_size); // All GUIDs are uint64_t
    uint8_t threadIdBytes = arm::pipe::numeric_cast<uint8_t>(ThreadIdSize);
    switch (threadIdBytes)
    {
    case 4: // Typically Windows and Android
    case 8: // Typically Linux
        break; // Valid values
    default:
        return TimelinePacketStatus::Error; // Invalid value
    }
    WriteUint8(buffer, offset, streamVersion);
    offset += uint8_t_size;
    WriteUint8(buffer, offset, pointerBytes);
    offset += uint8_t_size;
    WriteUint8(buffer, offset, threadIdBytes);
    offset += uint8_t_size;

    // Write the SWTrace directory
    uint32_t numberOfDeclarations = arm::pipe::numeric_cast<uint32_t>(timelineDirectoryMessages.size());
    WriteUint32(buffer, offset, numberOfDeclarations); // Number of declarations
    offset += uint32_t_size;
    for (uint32_t i : swTraceBuffer)
    {
        WriteUint32(buffer, offset, i); // Message declarations
        offset += uint32_t_size;
    }

    // Update the number of bytes written
    numberOfBytesWritten = timelineDirectoryPacketSize;

    return TimelinePacketStatus::Ok;
}

TimelinePacketStatus WriteTimelineEventClassBinary(uint64_t profilingGuid,
                                                   uint64_t nameGuid,
                                                   unsigned char* buffer,
                                                   unsigned int remainingBufferSize,
                                                   unsigned int& numberOfBytesWritten)
{
    // Initialize the output value
    numberOfBytesWritten = 0;

    // Check that the given buffer is valid
    if (buffer == nullptr || remainingBufferSize == 0)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // decl_id of the timeline message
    uint32_t declId = 2;

    // Calculate the length of the data (in bytes)
    unsigned int dataSize = uint32_t_size + (uint64_t_size * 2); // decl_id + Profiling GUID + Name GUID

    // Check whether the timeline binary fits in the given buffer
    if (dataSize > remainingBufferSize)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Initialize the offset for writing in the buffer
    unsigned int offset = 0;

    // Write the timeline binary payload to the buffer
    WriteUint32(buffer, offset, declId);        // decl_id
    offset += uint32_t_size;
    WriteUint64(buffer, offset, profilingGuid); // Profiling GUID
    offset += uint64_t_size;
    WriteUint64(buffer, offset, nameGuid); // Name GUID

    // Update the number of bytes written
    numberOfBytesWritten = dataSize;

    return TimelinePacketStatus::Ok;
}

TimelinePacketStatus WriteTimelineEventBinary(uint64_t timestamp,
                                              int threadId,
                                              uint64_t profilingGuid,
                                              unsigned char* buffer,
                                              unsigned int remainingBufferSize,
                                              unsigned int& numberOfBytesWritten)
{
    // Initialize the output value
    numberOfBytesWritten = 0;
    // Check that the given buffer is valid
    if (buffer == nullptr || remainingBufferSize == 0)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // decl_id of the timeline message
    uint32_t declId = 4;

    // Calculate the length of the data (in bytes)
    unsigned int timelineEventDataLength = uint32_t_size + // decl_id
                                           uint64_t_size + // Timestamp
                                           ThreadIdSize +  // Thread id
                                           uint64_t_size;  // Profiling GUID

    // Check whether the timeline binary packet fits in the given buffer
    if (timelineEventDataLength > remainingBufferSize)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Initialize the offset for writing in the buffer
    unsigned int offset = 0;

    // Write the timeline binary payload to the buffer
    WriteUint32(buffer, offset, declId); // decl_id
    offset += uint32_t_size;
    WriteUint64(buffer, offset, timestamp); // Timestamp
    offset += uint64_t_size;
    WriteBytes(buffer, offset, &threadId, ThreadIdSize); // Thread id
    offset += ThreadIdSize;
    WriteUint64(buffer, offset, profilingGuid); // Profiling GUID
    offset += uint64_t_size;
    // Update the number of bytes written
    numberOfBytesWritten = timelineEventDataLength;

    return TimelinePacketStatus::Ok;
}

uint64_t GetTimestamp()
{
    using clock = std::chrono::steady_clock;

    // Take a timestamp
    auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch());

    return static_cast<uint64_t>(timestamp.count());
}

arm::pipe::Packet ReceivePacket(const unsigned char* buffer, uint32_t length)
{
    if (buffer == nullptr)
    {
        throw arm::pipe::ProfilingException("data buffer is nullptr");
    }
    if (length < 8)
    {
        throw arm::pipe::ProfilingException("length of data buffer is less than 8");
    }

    uint32_t metadataIdentifier = 0;
    std::memcpy(&metadataIdentifier, buffer, sizeof(metadataIdentifier));

    uint32_t dataLength = 0;
    std::memcpy(&dataLength, buffer + 4u, sizeof(dataLength));

    std::unique_ptr<unsigned char[]> packetData;
    if (dataLength > 0)
    {
        packetData = std::make_unique<unsigned char[]>(dataLength);
        std::memcpy(packetData.get(), buffer + 8u, dataLength);
    }

    return arm::pipe::Packet(metadataIdentifier, dataLength, packetData);
}

} // namespace pipe

} // namespace arm

namespace std
{

bool operator==(const std::vector<uint8_t>& left, int right)
{
    return std::memcmp(left.data(), &right, left.size()) == 0;
}

} // namespace std
