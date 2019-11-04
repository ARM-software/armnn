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

void WriteBytes(const IPacketBufferPtr& packetBuffer, unsigned int offset,  const void* value, unsigned int valueSize)
{
    BOOST_ASSERT(packetBuffer);

    WriteBytes(packetBuffer->GetWritableData(), offset, value, valueSize);
}

void WriteUint64(const IPacketBufferPtr& packetBuffer, unsigned int offset, uint64_t value)
{
    BOOST_ASSERT(packetBuffer);

    WriteUint64(packetBuffer->GetWritableData(), offset, value);
}

void WriteUint32(const IPacketBufferPtr& packetBuffer, unsigned int offset, uint32_t value)
{
    BOOST_ASSERT(packetBuffer);

    WriteUint32(packetBuffer->GetWritableData(), offset, value);
}

void WriteUint16(const IPacketBufferPtr& packetBuffer, unsigned int offset, uint16_t value)
{
    BOOST_ASSERT(packetBuffer);

    WriteUint16(packetBuffer->GetWritableData(), offset, value);
}

void WriteBytes(unsigned char* buffer, unsigned int offset, const void* value, unsigned int valueSize)
{
    BOOST_ASSERT(buffer);
    BOOST_ASSERT(value);

    for (unsigned int i = 0; i < valueSize; i++, offset++)
    {
        buffer[offset] = *(reinterpret_cast<const unsigned char*>(value) + i);
    }
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

void ReadBytes(const IPacketBufferPtr& packetBuffer, unsigned int offset, unsigned int valueSize, uint8_t outValue[])
{
    BOOST_ASSERT(packetBuffer);

    ReadBytes(packetBuffer->GetReadableData(), offset, valueSize, outValue);
}

uint64_t ReadUint64(const IPacketBufferPtr& packetBuffer, unsigned int offset)
{
    BOOST_ASSERT(packetBuffer);

    return ReadUint64(packetBuffer->GetReadableData(), offset);
}

uint32_t ReadUint32(const IPacketBufferPtr& packetBuffer, unsigned int offset)
{
    BOOST_ASSERT(packetBuffer);

    return ReadUint32(packetBuffer->GetReadableData(), offset);
}

uint16_t ReadUint16(const IPacketBufferPtr& packetBuffer, unsigned int offset)
{
    BOOST_ASSERT(packetBuffer);

    return ReadUint16(packetBuffer->GetReadableData(), offset);
}

uint8_t ReadUint8(const IPacketBufferPtr& packetBuffer, unsigned int offset)
{
    BOOST_ASSERT(packetBuffer);

    return ReadUint8(packetBuffer->GetReadableData(), offset);
}

void ReadBytes(const unsigned char* buffer, unsigned int offset, unsigned int valueSize, uint8_t outValue[])
{
    BOOST_ASSERT(buffer);
    BOOST_ASSERT(outValue);

    for (unsigned int i = 0; i < valueSize; i++, offset++)
    {
        outValue[i] = static_cast<uint8_t>(buffer[offset]);
    }
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

// Calculate the actual length an SwString will be including the terminating null character
// padding to bring it to the next uint32_t boundary but minus the leading uint32_t encoding
// the size to allow the offset to be correctly updated when decoding a binary packet.
uint32_t CalculateSizeOfPaddedSwString(const std::string& str)
{
    std::vector<uint32_t> swTraceString;
    StringToSwTraceString<SwTraceCharPolicy>(str, swTraceString);
    unsigned int uint32_t_size = sizeof(uint32_t);
    uint32_t size = (boost::numeric_cast<uint32_t>(swTraceString.size()) - 1) * uint32_t_size;
    return size;
}

// Read TimelineMessageDirectoryPacket from given IPacketBuffer and offset
SwTraceMessage ReadSwTraceMessage(const IPacketBufferPtr& packetBuffer, unsigned int& offset)
{
    BOOST_ASSERT(packetBuffer);

    unsigned int uint32_t_size = sizeof(uint32_t);

    SwTraceMessage swTraceMessage;

    // Read the decl_id
    uint32_t readDeclId = ReadUint32(packetBuffer, offset);
    swTraceMessage.id = readDeclId;

    // SWTrace "namestring" format
    // length of the string (first 4 bytes) + string + null terminator

    // Check the decl_name
    offset += uint32_t_size;
    uint32_t swTraceDeclNameLength = ReadUint32(packetBuffer, offset);

    offset += uint32_t_size;
    std::vector<unsigned char> swTraceStringBuffer(swTraceDeclNameLength - 1);
    std::memcpy(swTraceStringBuffer.data(),
                packetBuffer->GetReadableData()  + offset, swTraceStringBuffer.size());

    swTraceMessage.name.assign(swTraceStringBuffer.begin(), swTraceStringBuffer.end()); // name

    // Check the ui_name
    offset += CalculateSizeOfPaddedSwString(swTraceMessage.name);
    uint32_t swTraceUINameLength = ReadUint32(packetBuffer, offset);

    offset += uint32_t_size;
    swTraceStringBuffer.resize(swTraceUINameLength - 1);
    std::memcpy(swTraceStringBuffer.data(),
                packetBuffer->GetReadableData()  + offset, swTraceStringBuffer.size());

    swTraceMessage.uiName.assign(swTraceStringBuffer.begin(), swTraceStringBuffer.end()); // ui_name

    // Check arg_types
    offset += CalculateSizeOfPaddedSwString(swTraceMessage.uiName);
    uint32_t swTraceArgTypesLength = ReadUint32(packetBuffer, offset);

    offset += uint32_t_size;
    swTraceStringBuffer.resize(swTraceArgTypesLength - 1);
    std::memcpy(swTraceStringBuffer.data(),
                packetBuffer->GetReadableData()  + offset, swTraceStringBuffer.size());

    swTraceMessage.argTypes.assign(swTraceStringBuffer.begin(), swTraceStringBuffer.end()); // arg_types

    std::string swTraceString(swTraceStringBuffer.begin(), swTraceStringBuffer.end());

    // Check arg_names
    offset += CalculateSizeOfPaddedSwString(swTraceString);
    uint32_t swTraceArgNamesLength = ReadUint32(packetBuffer, offset);

    offset += uint32_t_size;
    swTraceStringBuffer.resize(swTraceArgNamesLength - 1);
    std::memcpy(swTraceStringBuffer.data(),
                packetBuffer->GetReadableData()  + offset, swTraceStringBuffer.size());

    swTraceString.assign(swTraceStringBuffer.begin(), swTraceStringBuffer.end());
    std::stringstream stringStream(swTraceString);
    std::string argName;
    while (std::getline(stringStream, argName, ','))
    {
        swTraceMessage.argNames.push_back(argName);
    }

    offset += CalculateSizeOfPaddedSwString(swTraceString);

    return swTraceMessage;
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
                                                    unsigned int bufferSize,
                                                    unsigned int& numberOfBytesWritten)
{
    // Initialize the output value
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
    unsigned int timelineLabelPacketDataLength = uint32_t_size +   // decl_Id
                                                 uint64_t_size +   // Profiling GUID
                                                 swTraceLabelSize; // Label

    // Calculate the timeline binary packet size (in bytes)
    unsigned int timelineLabelPacketSize = 2 * uint32_t_size +            // Header (2 words)
                                           timelineLabelPacketDataLength; // decl_Id + Profiling GUID + label

    // Check whether the timeline binary packet fits in the given buffer
    if (timelineLabelPacketSize > bufferSize)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Create packet header
    std::pair<uint32_t, uint32_t> packetHeader = CreateTimelineMessagePacketHeader(timelineLabelPacketDataLength);

    // Initialize the offset for writing in the buffer
    unsigned int offset = 0;

    // Write the timeline binary packet header to the buffer
    WriteUint32(buffer, offset, packetHeader.first);
    offset += uint32_t_size;
    WriteUint32(buffer, offset, packetHeader.second);
    offset += uint32_t_size;

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
    numberOfBytesWritten = timelineLabelPacketSize;

    return TimelinePacketStatus::Ok;
}

TimelinePacketStatus WriteTimelineEntityBinaryPacket(uint64_t profilingGuid,
                                                     unsigned char* buffer,
                                                     unsigned int bufferSize,
                                                     unsigned int& numberOfBytesWritten)
{
    // Initialize the output value
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
                                            uint32_t_size +                 // decl_Id
                                            timelineEntityPacketDataLength; // Profiling GUID

    // Check whether the timeline binary packet fits in the given buffer
    if (timelineEntityPacketSize > bufferSize)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Create packet header
    std::pair<uint32_t, uint32_t> packetHeader = CreateTimelineMessagePacketHeader(timelineEntityPacketDataLength);

    // Initialize the offset for writing in the buffer
    unsigned int offset = 0;

    // Write the timeline binary packet header to the buffer
    WriteUint32(buffer, offset, packetHeader.first);
    offset += uint32_t_size;
    WriteUint32(buffer, offset, packetHeader.second);
    offset += uint32_t_size;

    // Write the decl_Id to the buffer
    WriteUint32(buffer, offset, 1u);
    offset += uint32_t_size;

    // Write the timeline binary packet payload to the buffer
    WriteUint64(buffer, offset, profilingGuid); // Profiling GUID

    // Update the number of bytes written
    numberOfBytesWritten = timelineEntityPacketSize;

    return TimelinePacketStatus::Ok;
}

TimelinePacketStatus WriteTimelineRelationshipBinaryPacket(ProfilingRelationshipType relationshipType,
                                                           uint64_t relationshipGuid,
                                                           uint64_t headGuid,
                                                           uint64_t tailGuid,
                                                           unsigned char* buffer,
                                                           unsigned int bufferSize,
                                                           unsigned int& numberOfBytesWritten)
{
    // Initialize the output value
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
    unsigned int timelineRelationshipPacketDataLength = uint32_t_size * 2 + // decl_id + Relationship Type
                                                        uint64_t_size * 3; // Relationship GUID + Head GUID + tail GUID

    // Calculate the timeline binary packet size (in bytes)
    unsigned int timelineRelationshipPacketSize = 2 * uint32_t_size + // Header (2 words)
                                                  timelineRelationshipPacketDataLength;

    // Check whether the timeline binary packet fits in the given buffer
    if (timelineRelationshipPacketSize > bufferSize)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Create packet header
    uint32_t dataLength = boost::numeric_cast<uint32_t>(timelineRelationshipPacketDataLength);
    std::pair<uint32_t, uint32_t> packetHeader = CreateTimelineMessagePacketHeader(dataLength);

    // Initialize the offset for writing in the buffer
    unsigned int offset = 0;

    // Write the timeline binary packet header to the buffer
    WriteUint32(buffer, offset, packetHeader.first);
    offset += uint32_t_size;
    WriteUint32(buffer, offset, packetHeader.second);
    offset += uint32_t_size;

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
            throw InvalidArgumentException("Unknown relationship type given.");
    }

    // Write the timeline binary packet payload to the buffer
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

    // Update the number of bytes written
    numberOfBytesWritten = timelineRelationshipPacketSize;

    return TimelinePacketStatus::Ok;
}

TimelinePacketStatus WriteTimelineMessageDirectoryPackage(unsigned char* buffer,
                                                          unsigned int bufferSize,
                                                          unsigned int& numberOfBytesWritten)
{
    // Initialize the output value
    numberOfBytesWritten = 0;

    // Check that the given buffer is valid
    if (buffer == nullptr || bufferSize == 0)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);

    // The payload/data of the packet consists of swtrace event definitions encoded according
    // to the swtrace directory specification. The messages being the five defined below:
    // |  decl_id  |  decl_name          |    ui_name            |  arg_types  |  arg_names                          |
    // |-----------|---------------------|-----------------------|-------------|-------------------------------------|
    // |    0      |   declareLabel      |   declare label       |    ps       |  guid,value                         |
    // |    1      |   declareEntity     |   declare entity      |    p        |  guid                               |
    // |    2      | declareEventClass   |  declare event class  |    p        |  guid                               |
    // |    3      | declareRelationship | declare relationship  |    Ippp     |  relationshipType,relationshipGuid, |
    // |           |                     |                       |             |  headGuid,tailGuid                  |
    // |    4      |   declareEvent      |   declare event       |    @tp      |  timestamp,threadId,eventGuid       |

    std::vector<std::vector<std::string>> timelineDirectoryMessages
    {
        {"declareLabel", "declare label", "ps", "guid,value"},
        {"declareEntity", "declare entity", "p", "guid"},
        {"declareEventClass", "declare event class", "p", "guid"},
        {"declareRelationship", "declare relationship", "Ippp", "relationshipType,relationshipGuid,headGuid,tailGuid"},
        {"declareEvent", "declare event", "@tp", "timestamp,threadId,eventGuid"}
    };

    unsigned int messagesDataLength = 0u;
    std::vector<std::vector<std::vector<uint32_t>>> swTraceTimelineDirectoryMessages;

    for (const auto& timelineDirectoryMessage : timelineDirectoryMessages)
    {
        messagesDataLength += uint32_t_size; // decl_id

        std::vector<std::vector<uint32_t>> swTraceStringsVector;
        for (const auto& label : timelineDirectoryMessage)
        {
            std::vector<uint32_t> swTraceString;
            bool result = StringToSwTraceString<SwTraceCharPolicy>(label, swTraceString);
            if (!result)
            {
                return TimelinePacketStatus::Error;
            }

            messagesDataLength += boost::numeric_cast<unsigned int>(swTraceString.size()) * uint32_t_size;
            swTraceStringsVector.push_back(swTraceString);
        }
        swTraceTimelineDirectoryMessages.push_back(swTraceStringsVector);
    }

    // Calculate the timeline directory binary packet size (in bytes)
    unsigned int timelineDirectoryPacketSize = 2 * uint32_t_size + // Header (2 words)
                                               messagesDataLength; // 5 messages length

    // Check whether the timeline directory binary packet fits in the given buffer
    if (timelineDirectoryPacketSize > bufferSize)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Create packet header
    uint32_t dataLength = boost::numeric_cast<uint32_t>(messagesDataLength);
    std::pair<uint32_t, uint32_t> packetHeader = CreateTimelinePacketHeader(1, 0, 0, 0, 0, dataLength);

    // Initialize the offset for writing in the buffer
    unsigned int offset = 0;

    // Write the timeline binary packet header to the buffer
    WriteUint32(buffer, offset, packetHeader.first);
    offset += uint32_t_size;
    WriteUint32(buffer, offset, packetHeader.second);
    offset += uint32_t_size;

    for (unsigned int i = 0u; i < swTraceTimelineDirectoryMessages.size(); ++i)
    {
        // Write the timeline binary packet payload to the buffer
        WriteUint32(buffer, offset, i); // decl_id
        offset += uint32_t_size;

        for (std::vector<uint32_t> swTraceString : swTraceTimelineDirectoryMessages[i])
        {
            for (uint32_t swTraceDeclStringWord : swTraceString)
            {
                WriteUint32(buffer, offset, swTraceDeclStringWord);
                offset += uint32_t_size;
            }
        }
    }

    // Update the number of bytes written
    numberOfBytesWritten = timelineDirectoryPacketSize;

    return TimelinePacketStatus::Ok;
}

TimelinePacketStatus WriteTimelineEventClassBinaryPacket(uint64_t profilingGuid,
                                                         unsigned char* buffer,
                                                         unsigned int bufferSize,
                                                         unsigned int& numberOfBytesWritten)
{
    // Initialize the output value
    numberOfBytesWritten = 0;

    // Check that the given buffer is valid
    if (buffer == nullptr || bufferSize == 0)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // decl_id of the timeline message
    uint32_t declId = 2;

    // Calculate the length of the data (in bytes)
    unsigned int packetBodySize = uint32_t_size + uint64_t_size; // decl_id + Profiling GUID

    // Calculate the timeline binary packet size (in bytes)
    unsigned int packetSize = 2 * uint32_t_size + // Header (2 words)
                              packetBodySize;     // Body

    // Check whether the timeline binary packet fits in the given buffer
    if (packetSize > bufferSize)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Create packet header
    std::pair<uint32_t, uint32_t> packetHeader = CreateTimelineMessagePacketHeader(packetBodySize);

    // Initialize the offset for writing in the buffer
    unsigned int offset = 0;

    // Write the timeline binary packet header to the buffer
    WriteUint32(buffer, offset, packetHeader.first);
    offset += uint32_t_size;
    WriteUint32(buffer, offset, packetHeader.second);
    offset += uint32_t_size;

    // Write the timeline binary packet payload to the buffer
    WriteUint32(buffer, offset, declId);        // decl_id
    offset += uint32_t_size;
    WriteUint64(buffer, offset, profilingGuid); // Profiling GUID

    // Update the number of bytes written
    numberOfBytesWritten = packetSize;

    return TimelinePacketStatus::Ok;
}

TimelinePacketStatus WriteTimelineEventBinaryPacket(uint64_t timestamp,
                                                    std::thread::id threadId,
                                                    uint64_t profilingGuid,
                                                    unsigned char* buffer,
                                                    unsigned int bufferSize,
                                                    unsigned int& numberOfBytesWritten)
{
    // Initialize the output value
    numberOfBytesWritten = 0;

    // Check that the given buffer is valid
    if (buffer == nullptr || bufferSize == 0)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Utils
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);
    unsigned int threadId_size = sizeof(std::thread::id);

    // decl_id of the timeline message
    uint32_t declId = 4;

    // Calculate the length of the data (in bytes)
    unsigned int timelineEventPacketDataLength = uint32_t_size + // decl_id
                                                 uint64_t_size + // Timestamp
                                                 threadId_size + // Thread id
                                                 uint64_t_size;  // Profiling GUID

    // Calculate the timeline binary packet size (in bytes)
    unsigned int timelineEventPacketSize = 2 * uint32_t_size +            // Header (2 words)
                                           timelineEventPacketDataLength; // Timestamp + thread id + profiling GUID

    // Check whether the timeline binary packet fits in the given buffer
    if (timelineEventPacketSize > bufferSize)
    {
        return TimelinePacketStatus::BufferExhaustion;
    }

    // Create packet header
    std::pair<uint32_t, uint32_t> packetHeader = CreateTimelineMessagePacketHeader(timelineEventPacketDataLength);

    // Initialize the offset for writing in the buffer
    unsigned int offset = 0;

    // Write the timeline binary packet header to the buffer
    WriteUint32(buffer, offset, packetHeader.first);
    offset += uint32_t_size;
    WriteUint32(buffer, offset, packetHeader.second);
    offset += uint32_t_size;

    // Write the timeline binary packet payload to the buffer
    WriteUint32(buffer, offset, declId); // decl_id
    offset += uint32_t_size;
    WriteUint64(buffer, offset, timestamp); // Timestamp
    offset += uint64_t_size;
    WriteBytes(buffer, offset, &threadId, threadId_size); // Thread id
    offset += threadId_size;
    WriteUint64(buffer, offset, profilingGuid); // Profiling GUID
    offset += uint64_t_size;

    // Update the number of bytes written
    numberOfBytesWritten = timelineEventPacketSize;

    return TimelinePacketStatus::Ok;
}

} // namespace profiling

} // namespace armnn

namespace std
{

bool operator==(const std::vector<uint8_t>& left, std::thread::id right)
{
    return std::memcmp(left.data(), &right, left.size()) == 0;
}

} // namespace std
