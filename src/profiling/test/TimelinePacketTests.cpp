//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <ProfilingUtils.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/numeric/conversion/cast.hpp>

using namespace armnn::profiling;

BOOST_AUTO_TEST_SUITE(TimelinePacketTests)

BOOST_AUTO_TEST_CASE(TimelineLabelPacketTest1)
{
    const uint64_t profilingGuid = 123456u;
    const std::string label = "some label";
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineLabelBinaryPacket(profilingGuid,
                                                                 label,
                                                                 nullptr,
                                                                 512u,
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineLabelPacketTest2)
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    const std::string label = "some label";
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineLabelBinaryPacket(profilingGuid,
                                                                 label,
                                                                 buffer.data(),
                                                                 0,
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineLabelPacketTest3)
{
    std::vector<unsigned char> buffer(10, 0);

    const uint64_t profilingGuid = 123456u;
    const std::string label = "some label";
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineLabelBinaryPacket(profilingGuid,
                                                                 label,
                                                                 buffer.data(),
                                                                 boost::numeric_cast<unsigned int>(buffer.size()),
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineLabelPacketTest4)
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    const std::string label = "s0m€ l@b€l";
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineLabelBinaryPacket(profilingGuid,
                                                                 label,
                                                                 buffer.data(),
                                                                 boost::numeric_cast<unsigned int>(buffer.size()),
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::Error);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineLabelPacketTest5)
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    const std::string label = "some label";
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineLabelBinaryPacket(profilingGuid,
                                                                 label,
                                                                 buffer.data(),
                                                                 boost::numeric_cast<unsigned int>(buffer.size()),
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::Ok);
    BOOST_CHECK(numberOfBytesWritten == 36);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the packet header
    unsigned int offset = 0;
    uint32_t packetHeaderWord0 = ReadUint32(buffer.data(), offset);
    uint32_t packetFamily = (packetHeaderWord0 >> 26) & 0x0000003F;
    uint32_t packetClass  = (packetHeaderWord0 >> 19) & 0x0000007F;
    uint32_t packetType   = (packetHeaderWord0 >> 16) & 0x00000007;
    uint32_t streamId     = (packetHeaderWord0 >>  0) & 0x00000007;
    BOOST_CHECK(packetFamily == 1);
    BOOST_CHECK(packetClass  == 0);
    BOOST_CHECK(packetType   == 1);
    BOOST_CHECK(streamId     == 0);

    offset += uint32_t_size;
    uint32_t packetHeaderWord1 = ReadUint32(buffer.data(), offset);
    uint32_t sequenceNumbered = (packetHeaderWord1 >> 24) & 0x00000001;
    uint32_t dataLength       = (packetHeaderWord1 >>  0) & 0x00FFFFFF;
    BOOST_CHECK(sequenceNumbered ==  0);
    BOOST_CHECK(dataLength       == 28);

    // Check decl_Id
    offset += uint32_t_size;
    uint32_t decl_Id = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(decl_Id == uint32_t(0));

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readProfilingGuid == profilingGuid);

    // Check the SWTrace label
    offset += uint64_t_size;
    uint32_t swTraceLabelLength = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(swTraceLabelLength == 11); // Label length including the null-terminator

    offset += uint32_t_size;
    BOOST_CHECK(std::memcmp(buffer.data() + offset,        // Offset to the label in the buffer
                            label.data(),                  // The original label
                            swTraceLabelLength - 1) == 0); // The length of the label

    offset += swTraceLabelLength * uint32_t_size;
    BOOST_CHECK(buffer[offset] == '\0'); // The null-terminator at the end of the SWTrace label
}

BOOST_AUTO_TEST_CASE(TimelineRelationshipPacketNullBufferTest)
{
    ProfilingRelationshipType relationshipType = ProfilingRelationshipType::DataLink;
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineRelationshipBinaryPacket(relationshipType,
                                                                        relationshipGuid,
                                                                        headGuid,
                                                                        tailGuid,
                                                                        nullptr,
                                                                        512u,
                                                                        numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineRelationshipPacketZeroBufferSizeTest)
{
    std::vector<unsigned char> buffer(512, 0);

    ProfilingRelationshipType relationshipType = ProfilingRelationshipType::DataLink;
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineRelationshipBinaryPacket(relationshipType,
                                                                        relationshipGuid,
                                                                        headGuid,
                                                                        tailGuid,
                                                                        buffer.data(),
                                                                        0,
                                                                        numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineRelationshipPacketSmallBufferSizeTest)
{
    std::vector<unsigned char> buffer(10, 0);

    ProfilingRelationshipType relationshipType = ProfilingRelationshipType::DataLink;
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result =
        WriteTimelineRelationshipBinaryPacket(relationshipType,
                                              relationshipGuid,
                                              headGuid,
                                              tailGuid,
                                              buffer.data(),
                                              boost::numeric_cast<unsigned int>(buffer.size()),
                                              numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineRelationshipPacketInvalidRelationTest)
{
    std::vector<unsigned char> buffer(512, 0);
    ProfilingRelationshipType relationshipType = static_cast<ProfilingRelationshipType>(5);
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    unsigned int numberOfBytesWritten = 789u;

    BOOST_CHECK_THROW(WriteTimelineRelationshipBinaryPacket(relationshipType,
                                                            relationshipGuid,
                                                            headGuid,
                                                            tailGuid,
                                                            buffer.data(),
                                                            boost::numeric_cast<unsigned int>(buffer.size()),
                                                            numberOfBytesWritten),
                       armnn::InvalidArgumentException);

    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineRelationshipPacketTest)
{
    std::vector<unsigned char> buffer(512, 0);

    ProfilingRelationshipType relationshipType = ProfilingRelationshipType::RetentionLink;
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result =
        WriteTimelineRelationshipBinaryPacket(relationshipType,
                                              relationshipGuid,
                                              headGuid,
                                              tailGuid,
                                              buffer.data(),
                                              boost::numeric_cast<unsigned int>(buffer.size()),
                                              numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::Ok);
    BOOST_CHECK(numberOfBytesWritten == 40);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the packet header
    unsigned int offset = 0;
    uint32_t packetHeaderWord0 = ReadUint32(buffer.data(), offset);
    uint32_t packetFamily = (packetHeaderWord0 >> 26) & 0x0000003F;
    uint32_t packetClass  = (packetHeaderWord0 >> 19) & 0x0000007F;
    uint32_t packetType   = (packetHeaderWord0 >> 16) & 0x00000007;
    uint32_t streamId     = (packetHeaderWord0 >>  0) & 0x00000007;
    BOOST_CHECK(packetFamily == 1);
    BOOST_CHECK(packetClass  == 0);
    BOOST_CHECK(packetType   == 1);
    BOOST_CHECK(streamId     == 0);

    offset += uint32_t_size;
    uint32_t packetHeaderWord1 = ReadUint32(buffer.data(), offset);
    uint32_t sequenceNumbered = (packetHeaderWord1 >> 24) & 0x00000001;
    uint32_t dataLength       = (packetHeaderWord1 >>  0) & 0x00FFFFFF;
    BOOST_CHECK(sequenceNumbered ==  0);
    BOOST_CHECK(dataLength       == 32);

    // Check the decl_id
    offset += uint32_t_size;
    uint32_t readDeclId = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(readDeclId == 3);

    // Check the relationship type
    offset += uint32_t_size;
    uint32_t readRelationshipType = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(readRelationshipType == 0);

    // Check the relationship GUID
    offset += uint32_t_size;
    uint64_t readRelationshipGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readRelationshipGuid == relationshipGuid);

    // Check the head GUID
    offset += uint64_t_size;
    uint64_t readHeadGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readHeadGuid == headGuid);

    // Check the tail GUID
    offset += uint64_t_size;
    uint64_t readTailGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readTailGuid == tailGuid);
}

BOOST_AUTO_TEST_CASE(TimelineRelationshipPacketExecutionLinkTest)
{
    std::vector<unsigned char> buffer(512, 0);

    ProfilingRelationshipType relationshipType = ProfilingRelationshipType::ExecutionLink;
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result =
        WriteTimelineRelationshipBinaryPacket(relationshipType,
                                              relationshipGuid,
                                              headGuid,
                                              tailGuid,
                                              buffer.data(),
                                              boost::numeric_cast<unsigned int>(buffer.size()),
                                              numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::Ok);
    BOOST_CHECK(numberOfBytesWritten == 40);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the packet header
    unsigned int offset = 0;
    uint32_t packetHeaderWord0 = ReadUint32(buffer.data(), offset);
    uint32_t packetFamily = (packetHeaderWord0 >> 26) & 0x0000003F;
    uint32_t packetClass  = (packetHeaderWord0 >> 19) & 0x0000007F;
    uint32_t packetType   = (packetHeaderWord0 >> 16) & 0x00000007;
    uint32_t streamId     = (packetHeaderWord0 >>  0) & 0x00000007;
    BOOST_CHECK(packetFamily == 1);
    BOOST_CHECK(packetClass  == 0);
    BOOST_CHECK(packetType   == 1);
    BOOST_CHECK(streamId     == 0);

    offset += uint32_t_size;
    uint32_t packetHeaderWord1 = ReadUint32(buffer.data(), offset);
    uint32_t sequenceNumbered = (packetHeaderWord1 >> 24) & 0x00000001;
    uint32_t dataLength       = (packetHeaderWord1 >>  0) & 0x00FFFFFF;
    BOOST_CHECK(sequenceNumbered ==  0);
    BOOST_CHECK(dataLength       == 32);

    // Check the decl_id
    offset += uint32_t_size;
    uint32_t readDeclId = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(readDeclId == 3);

    // Check the relationship type
    offset += uint32_t_size;
    uint32_t readRelationshipType = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(readRelationshipType == 1);

    // Check the relationship GUID
    offset += uint32_t_size;
    uint64_t readRelationshipGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readRelationshipGuid == relationshipGuid);

    // Check the head GUID
    offset += uint64_t_size;
    uint64_t readHeadGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readHeadGuid == headGuid);

    // Check the tail GUID
    offset += uint64_t_size;
    uint64_t readTailGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readTailGuid == tailGuid);
}


BOOST_AUTO_TEST_CASE(TimelineRelationshipPacketDataLinkTest)
{
    std::vector<unsigned char> buffer(512, 0);

    ProfilingRelationshipType relationshipType = ProfilingRelationshipType::DataLink;
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result =
        WriteTimelineRelationshipBinaryPacket(relationshipType,
                                              relationshipGuid,
                                              headGuid,
                                              tailGuid,
                                              buffer.data(),
                                              boost::numeric_cast<unsigned int>(buffer.size()),
                                              numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::Ok);
    BOOST_CHECK(numberOfBytesWritten == 40);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the packet header
    unsigned int offset = 0;
    uint32_t packetHeaderWord0 = ReadUint32(buffer.data(), offset);
    uint32_t packetFamily = (packetHeaderWord0 >> 26) & 0x0000003F;
    uint32_t packetClass  = (packetHeaderWord0 >> 19) & 0x0000007F;
    uint32_t packetType   = (packetHeaderWord0 >> 16) & 0x00000007;
    uint32_t streamId     = (packetHeaderWord0 >>  0) & 0x00000007;
    BOOST_CHECK(packetFamily == 1);
    BOOST_CHECK(packetClass  == 0);
    BOOST_CHECK(packetType   == 1);
    BOOST_CHECK(streamId     == 0);

    offset += uint32_t_size;
    uint32_t packetHeaderWord1 = ReadUint32(buffer.data(), offset);
    uint32_t sequenceNumbered = (packetHeaderWord1 >> 24) & 0x00000001;
    uint32_t dataLength       = (packetHeaderWord1 >>  0) & 0x00FFFFFF;
    BOOST_CHECK(sequenceNumbered ==  0);
    BOOST_CHECK(dataLength       == 32);

    // Check the decl_id
    offset += uint32_t_size;
    uint32_t readDeclId = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(readDeclId == 3);

    // Check the relationship type
    offset += uint32_t_size;
    uint32_t readRelationshipType = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(readRelationshipType == 2);

    // Check the relationship GUID
    offset += uint32_t_size;
    uint64_t readRelationshipGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readRelationshipGuid == relationshipGuid);

    // Check the head GUID
    offset += uint64_t_size;
    uint64_t readHeadGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readHeadGuid == headGuid);

    // Check the tail GUID
    offset += uint64_t_size;
    uint64_t readTailGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readTailGuid == tailGuid);
}


BOOST_AUTO_TEST_CASE(TimelineRelationshipPacketLabelLinkTest)
{
    std::vector<unsigned char> buffer(512, 0);

    ProfilingRelationshipType relationshipType = ProfilingRelationshipType::LabelLink;
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result =
        WriteTimelineRelationshipBinaryPacket(relationshipType,
                                              relationshipGuid,
                                              headGuid,
                                              tailGuid,
                                              buffer.data(),
                                              boost::numeric_cast<unsigned int>(buffer.size()),
                                              numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::Ok);
    BOOST_CHECK(numberOfBytesWritten == 40);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the packet header
    unsigned int offset = 0;
    uint32_t packetHeaderWord0 = ReadUint32(buffer.data(), offset);
    uint32_t packetFamily = (packetHeaderWord0 >> 26) & 0x0000003F;
    uint32_t packetClass  = (packetHeaderWord0 >> 19) & 0x0000007F;
    uint32_t packetType   = (packetHeaderWord0 >> 16) & 0x00000007;
    uint32_t streamId     = (packetHeaderWord0 >>  0) & 0x00000007;
    BOOST_CHECK(packetFamily == 1);
    BOOST_CHECK(packetClass  == 0);
    BOOST_CHECK(packetType   == 1);
    BOOST_CHECK(streamId     == 0);

    offset += uint32_t_size;
    uint32_t packetHeaderWord1 = ReadUint32(buffer.data(), offset);
    uint32_t sequenceNumbered = (packetHeaderWord1 >> 24) & 0x00000001;
    uint32_t dataLength       = (packetHeaderWord1 >>  0) & 0x00FFFFFF;
    BOOST_CHECK(sequenceNumbered ==  0);
    BOOST_CHECK(dataLength       == 32);

    // Check the decl_id
    offset += uint32_t_size;
    uint32_t readDeclId = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(readDeclId == 3);

    // Check the relationship type
    offset += uint32_t_size;
    uint32_t readRelationshipType = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(readRelationshipType == 3);

    // Check the relationship GUID
    offset += uint32_t_size;
    uint64_t readRelationshipGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readRelationshipGuid == relationshipGuid);

    // Check the head GUID
    offset += uint64_t_size;
    uint64_t readHeadGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readHeadGuid == headGuid);

    // Check the tail GUID
    offset += uint64_t_size;
    uint64_t readTailGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readTailGuid == tailGuid);
}

BOOST_AUTO_TEST_CASE(TimelineMessageDirectoryPacketTest1)
{
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineMessageDirectoryPackage(nullptr,
                                                                       512u,
                                                                       numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineMessageDirectoryPacketTest2)
{
    std::vector<unsigned char> buffer(512, 0);

    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineMessageDirectoryPackage(buffer.data(),
                                                                       0,
                                                                       numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineMessageDirectoryPacketTest3)
{
    std::vector<unsigned char> buffer(512, 0);
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineMessageDirectoryPackage(buffer.data(),
                                                                       boost::numeric_cast<unsigned int>(buffer.size()),
                                                                       numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::Ok);

    BOOST_CHECK(numberOfBytesWritten == 427);

    unsigned int uint8_t_size  = sizeof(uint8_t);
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);
    unsigned int threadId_size = sizeof(std::thread::id);

    // Check the packet header
    unsigned int offset = 0;
    uint32_t packetHeaderWord0 = ReadUint32(buffer.data(), offset);
    uint32_t packetFamily = (packetHeaderWord0 >> 26) & 0x0000003F;
    uint32_t packetClass  = (packetHeaderWord0 >> 19) & 0x0000007F;
    uint32_t packetType   = (packetHeaderWord0 >> 16) & 0x00000007;
    uint32_t streamId     = (packetHeaderWord0 >>  0) & 0x00000007;
    BOOST_CHECK(packetFamily == 1);
    BOOST_CHECK(packetClass  == 0);
    BOOST_CHECK(packetType   == 0);
    BOOST_CHECK(streamId     == 0);

    offset += uint32_t_size;
    uint32_t packetHeaderWord1 = ReadUint32(buffer.data(), offset);
    uint32_t sequenceNumbered = (packetHeaderWord1 >> 24) & 0x00000001;
    uint32_t dataLength       = (packetHeaderWord1 >>  0) & 0x00FFFFFF;
    BOOST_CHECK(sequenceNumbered ==  0);
    BOOST_CHECK(dataLength       == 419);

    // Check the stream header
    offset += uint32_t_size;
    uint8_t readStreamVersion = ReadUint8(buffer.data(), offset);
    BOOST_CHECK(readStreamVersion == 4);
    offset += uint8_t_size;
    uint8_t readPointerBytes = ReadUint8(buffer.data(), offset);
    BOOST_CHECK(readPointerBytes == uint64_t_size);
    offset += uint8_t_size;
    uint8_t readThreadIdBytes = ReadUint8(buffer.data(), offset);
    BOOST_CHECK(readThreadIdBytes == threadId_size);

    // Check the number of declarations
    offset += uint8_t_size;
    uint32_t declCount = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(declCount == 5);

    // Check the decl_id
    offset += uint32_t_size;
    uint32_t readDeclId = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(readDeclId == 0);

    // SWTrace "namestring" format
    // length of the string (first 4 bytes) + string + null terminator

    // Check the decl_name
    offset += uint32_t_size;
    uint32_t swTraceDeclNameLength = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(swTraceDeclNameLength == 13); // decl_name length including the null-terminator

    std::string label = "declareLabel";
    offset += uint32_t_size;
    BOOST_CHECK(std::memcmp(buffer.data() + offset,           // Offset to the label in the buffer
                            label.data(),                     // The original label
                            swTraceDeclNameLength - 1) == 0); // The length of the label

    // Check the ui_name
    std::vector<uint32_t> swTraceString;
    StringToSwTraceString<SwTraceCharPolicy>(label, swTraceString);
    offset += (boost::numeric_cast<unsigned int>(swTraceString.size()) - 1) * uint32_t_size;
    uint32_t swTraceUINameLength = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(swTraceUINameLength == 14); // ui_name length including the null-terminator

    label = "declare label";
    offset += uint32_t_size;
    BOOST_CHECK(std::memcmp(buffer.data() + offset,           // Offset to the label in the buffer
                            label.data(),                     // The original label
                            swTraceUINameLength - 1) == 0);   // The length of the label

    // Check arg_types
    StringToSwTraceString<SwTraceCharPolicy>(label, swTraceString);
    offset += (boost::numeric_cast<unsigned int>(swTraceString.size()) - 1) * uint32_t_size;
    uint32_t swTraceArgTypesLength = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(swTraceArgTypesLength == 3); // arg_types length including the null-terminator

    label = "ps";
    offset += uint32_t_size;
    BOOST_CHECK(std::memcmp(buffer.data() + offset,           // Offset to the label in the buffer
                            label.data(),                     // The original label
                            swTraceArgTypesLength - 1) == 0); // The length of the label

    // Check arg_names
    StringToSwTraceString<SwTraceCharPolicy>(label, swTraceString);
    offset += (boost::numeric_cast<unsigned int>(swTraceString.size()) - 1) * uint32_t_size;
    uint32_t swTraceArgNamesLength = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(swTraceArgNamesLength == 11); // arg_names length including the null-terminator

    label = "guid,value";
    offset += uint32_t_size;
    BOOST_CHECK(std::memcmp(buffer.data() + offset,        // Offset to the label in the buffer
                            label.data(),                     // The original label
                            swTraceArgNamesLength - 1) == 0); // The length of the label

    // Check second message decl_id
    StringToSwTraceString<SwTraceCharPolicy>(label, swTraceString);
    offset += (boost::numeric_cast<unsigned int>(swTraceString.size()) - 1) * uint32_t_size;
    readDeclId = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(readDeclId == 1);

    // Check second decl_name
    offset += uint32_t_size;
    swTraceDeclNameLength = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(swTraceDeclNameLength == 14); // decl_name length including the null-terminator

    label = "declareEntity";
    offset += uint32_t_size;
    BOOST_CHECK(std::memcmp(buffer.data() + offset,           // Offset to the label in the buffer
                            label.data(),                     // The original label
                            swTraceDeclNameLength - 1) == 0); // The length of the label
}

BOOST_AUTO_TEST_CASE(TimelineEntityPacketTest1)
{
    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEntityBinaryPacket(profilingGuid,
                                                                 nullptr,
                                                                 512u,
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineEntityPacketTest2)
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEntityBinaryPacket(profilingGuid,
                                                                 buffer.data(),
                                                                 0,
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineEntityPacketTest3)
{
    std::vector<unsigned char> buffer(10, 0);

    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEntityBinaryPacket(profilingGuid,
                                                                 buffer.data(),
                                                                 boost::numeric_cast<unsigned int>(buffer.size()),
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineEntityPacketTest4)
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEntityBinaryPacket(profilingGuid,
                                                                 buffer.data(),
                                                                 boost::numeric_cast<unsigned int>(buffer.size()),
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::Ok);
    BOOST_CHECK(numberOfBytesWritten == 20);

    unsigned int uint32_t_size = sizeof(uint32_t);

    // Check the packet header
    unsigned int offset = 0;
    uint32_t packetHeaderWord0 = ReadUint32(buffer.data(), offset);
    uint32_t packetFamily = (packetHeaderWord0 >> 26) & 0x0000003F;
    uint32_t packetClass  = (packetHeaderWord0 >> 19) & 0x0000007F;
    uint32_t packetType   = (packetHeaderWord0 >> 16) & 0x00000007;
    uint32_t streamId     = (packetHeaderWord0 >>  0) & 0x00000007;
    BOOST_CHECK(packetFamily == 1);
    BOOST_CHECK(packetClass  == 0);
    BOOST_CHECK(packetType   == 1);
    BOOST_CHECK(streamId     == 0);

    offset += uint32_t_size;
    uint32_t packetHeaderWord1 = ReadUint32(buffer.data(), offset);
    uint32_t sequenceNumbered = (packetHeaderWord1 >> 24) & 0x00000001;
    uint32_t dataLength       = (packetHeaderWord1 >>  0) & 0x00FFFFFF;
    BOOST_CHECK(sequenceNumbered ==  0);
    BOOST_CHECK(dataLength       == 12);

    // Check decl_Id
    offset += uint32_t_size;
    uint32_t decl_Id = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(decl_Id == uint32_t(1));

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readProfilingGuid == profilingGuid);
}

BOOST_AUTO_TEST_CASE(TimelineEventClassTest1)
{
    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventClassBinaryPacket(profilingGuid,
                                                                      nullptr,
                                                                      512u,
                                                                      numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineEventClassTest2)
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventClassBinaryPacket(profilingGuid,
                                                                      buffer.data(),
                                                                      0,
                                                                      numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineEventClassTest3)
{
    std::vector<unsigned char> buffer(10, 0);

    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventClassBinaryPacket(profilingGuid,
                                                                      buffer.data(),
                                                                      boost::numeric_cast<unsigned int>(buffer.size()),
                                                                      numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineEventClassTest4)
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventClassBinaryPacket(profilingGuid,
                                                                      buffer.data(),
                                                                      boost::numeric_cast<unsigned int>(buffer.size()),
                                                                      numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::Ok);
    BOOST_CHECK(numberOfBytesWritten == 20);

    unsigned int uint32_t_size = sizeof(uint32_t);

    // Check the packet header
    unsigned int offset = 0;
    uint32_t packetHeaderWord0 = ReadUint32(buffer.data(), offset);
    uint32_t packetFamily = (packetHeaderWord0 >> 26) & 0x0000003F;
    uint32_t packetClass  = (packetHeaderWord0 >> 19) & 0x0000007F;
    uint32_t packetType   = (packetHeaderWord0 >> 16) & 0x00000007;
    uint32_t streamId     = (packetHeaderWord0 >>  0) & 0x00000007;
    BOOST_CHECK(packetFamily == 1);
    BOOST_CHECK(packetClass  == 0);
    BOOST_CHECK(packetType   == 1);
    BOOST_CHECK(streamId     == 0);

    offset += uint32_t_size;
    uint32_t packetHeaderWord1 = ReadUint32(buffer.data(), offset);
    uint32_t sequenceNumbered = (packetHeaderWord1 >> 24) & 0x00000001;
    uint32_t dataLength       = (packetHeaderWord1 >>  0) & 0x00FFFFFF;
    BOOST_CHECK(sequenceNumbered ==  0);
    BOOST_CHECK(dataLength       == 12);

    // Check the decl_id
    offset += uint32_t_size;
    uint32_t declId = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(declId == uint32_t(2));

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readProfilingGuid == profilingGuid);
}

BOOST_AUTO_TEST_CASE(TimelineEventPacketTest1)
{
    const uint64_t timestamp = 456789u;
    const std::thread::id threadId = std::this_thread::get_id();
    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventBinaryPacket(timestamp,
                                                                 threadId,
                                                                 profilingGuid,
                                                                 nullptr,
                                                                 512u,
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineEventPacketTest2)
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t timestamp = 456789u;
    const std::thread::id threadId = std::this_thread::get_id();
    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventBinaryPacket(timestamp,
                                                                 threadId,
                                                                 profilingGuid,
                                                                 buffer.data(),
                                                                 0,
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineEventPacketTest3)
{
    std::vector<unsigned char> buffer(10, 0);

    const uint64_t timestamp = 456789u;
    const std::thread::id threadId = std::this_thread::get_id();
    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventBinaryPacket(timestamp,
                                                                 threadId,
                                                                 profilingGuid,
                                                                 buffer.data(),
                                                                 boost::numeric_cast<unsigned int>(buffer.size()),
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::BufferExhaustion);
    BOOST_CHECK(numberOfBytesWritten == 0);
}

BOOST_AUTO_TEST_CASE(TimelineEventPacketTest4)
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t timestamp = 456789u;
    const std::thread::id threadId = std::this_thread::get_id();
    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventBinaryPacket(timestamp,
                                                                 threadId,
                                                                 profilingGuid,
                                                                 buffer.data(),
                                                                 boost::numeric_cast<unsigned int>(buffer.size()),
                                                                 numberOfBytesWritten);
    BOOST_CHECK(result == TimelinePacketStatus::Ok);
    BOOST_CHECK(numberOfBytesWritten == 36);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);
    unsigned int threadId_size = sizeof(std::thread::id);

    // Check the packet header
    unsigned int offset = 0;
    uint32_t packetHeaderWord0 = ReadUint32(buffer.data(), offset);
    uint32_t packetFamily = (packetHeaderWord0 >> 26) & 0x0000003F;
    uint32_t packetClass  = (packetHeaderWord0 >> 19) & 0x0000007F;
    uint32_t packetType   = (packetHeaderWord0 >> 16) & 0x00000007;
    uint32_t streamId     = (packetHeaderWord0 >>  0) & 0x00000007;
    BOOST_CHECK(packetFamily == 1);
    BOOST_CHECK(packetClass  == 0);
    BOOST_CHECK(packetType   == 1);
    BOOST_CHECK(streamId     == 0);

    offset += uint32_t_size;
    uint32_t packetHeaderWord1 = ReadUint32(buffer.data(), offset);
    uint32_t sequenceNumbered = (packetHeaderWord1 >> 24) & 0x00000001;
    uint32_t dataLength       = (packetHeaderWord1 >>  0) & 0x00FFFFFF;
    BOOST_CHECK(sequenceNumbered ==  0);
    BOOST_CHECK(dataLength       == 28);

    // Check the decl_id
    offset += uint32_t_size;
    uint32_t readDeclId = ReadUint32(buffer.data(), offset);
    BOOST_CHECK(readDeclId == 4);

    // Check the timestamp
    offset += uint32_t_size;
    uint64_t readTimestamp = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readTimestamp == timestamp);

    // Check the thread id
    offset += uint64_t_size;
    std::vector<uint8_t> readThreadId(threadId_size, 0);
    ReadBytes(buffer.data(), offset, threadId_size, readThreadId.data());
    BOOST_CHECK(readThreadId == threadId);

    // Check the profiling GUID
    offset += threadId_size;
    uint64_t readProfilingGuid = ReadUint64(buffer.data(), offset);
    BOOST_CHECK(readProfilingGuid == profilingGuid);
}

BOOST_AUTO_TEST_SUITE_END()
