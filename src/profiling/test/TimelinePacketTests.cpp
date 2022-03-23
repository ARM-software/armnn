//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <client/src/ProfilingUtils.hpp>

#include <common/include/NumericCast.hpp>
#include <common/include/SwTrace.hpp>
#include <common/include/Threads.hpp>

#include <doctest/doctest.h>

using namespace arm::pipe;

TEST_SUITE("TimelinePacketTests")
{
TEST_CASE("TimelineLabelPacketTestNoBuffer")
{
    const uint64_t profilingGuid = 123456u;
    const std::string label = "some label";
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineLabelBinaryPacket(profilingGuid,
                                                                 label,
                                                                 nullptr,
                                                                 512u,
                                                                 numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineLabelPacketTestBufferExhaustionZeroValue")
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
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineLabelPacketTestBufferExhaustionFixedValue")
{
    std::vector<unsigned char> buffer(10, 0);

    const uint64_t profilingGuid = 123456u;
    const std::string label = "some label";
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineLabelBinaryPacket(profilingGuid,
                                                                 label,
                                                                 buffer.data(),
                                                                 arm::pipe::numeric_cast<unsigned int>(buffer.size()),
                                                                 numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineLabelPacketTestInvalidLabel")
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    const std::string label = "s0m€ l@b€l";
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineLabelBinaryPacket(profilingGuid,
                                                                 label,
                                                                 buffer.data(),
                                                                 arm::pipe::numeric_cast<unsigned int>(buffer.size()),
                                                                 numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::Error);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineLabelPacketTestSingleConstructionOfData")
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    const std::string label = "some label";
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineLabelBinaryPacket(profilingGuid,
                                                                 label,
                                                                 buffer.data(),
                                                                 arm::pipe::numeric_cast<unsigned int>(buffer.size()),
                                                                 numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::Ok);
    CHECK(numberOfBytesWritten == 28);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the packet header
    unsigned int offset = 0;
    uint32_t decl_Id = ReadUint32(buffer.data(), offset);
    CHECK(decl_Id == uint32_t(0));

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(buffer.data(), offset);
    CHECK(readProfilingGuid == profilingGuid);

    // Check the SWTrace label
    offset += uint64_t_size;
    uint32_t swTraceLabelLength = ReadUint32(buffer.data(), offset);
    CHECK(swTraceLabelLength == 11); // Label length including the null-terminator

    offset += uint32_t_size;
    CHECK(std::memcmp(buffer.data() + offset,        // Offset to the label in the buffer
                            label.data(),                  // The original label
                            swTraceLabelLength - 1) == 0); // The length of the label

    offset += swTraceLabelLength * uint32_t_size;
    CHECK(buffer[offset] == '\0'); // The null-terminator at the end of the SWTrace label
}

TEST_CASE("TimelineRelationshipPacketNullBufferTest")
{
    ProfilingRelationshipType relationshipType = ProfilingRelationshipType::DataLink;
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    const uint64_t attributeGuid = 876345u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineRelationshipBinary(relationshipType,
                                                                  relationshipGuid,
                                                                  headGuid,
                                                                  tailGuid,
                                                                  attributeGuid,
                                                                  nullptr,
                                                                  512u,
                                                                  numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineRelationshipPacketZeroBufferSizeTest")
{
    std::vector<unsigned char> buffer(512, 0);

    ProfilingRelationshipType relationshipType = ProfilingRelationshipType::DataLink;
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    const uint64_t attributeGuid = 876345u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineRelationshipBinary(relationshipType,
                                                                  relationshipGuid,
                                                                  headGuid,
                                                                  tailGuid,
                                                                  attributeGuid,
                                                                  buffer.data(),
                                                                  0,
                                                                  numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineRelationshipPacketSmallBufferSizeTest")
{
    std::vector<unsigned char> buffer(10, 0);

    ProfilingRelationshipType relationshipType = ProfilingRelationshipType::DataLink;
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    const uint64_t attributeGuid = 876345u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result =
                             WriteTimelineRelationshipBinary(relationshipType,
                                                             relationshipGuid,
                                                             headGuid,
                                                             tailGuid,
                                                             attributeGuid,
                                                             buffer.data(),
                                                             arm::pipe::numeric_cast<unsigned int>(buffer.size()),
                                                             numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineRelationshipPacketInvalidRelationTest")
{
    std::vector<unsigned char> buffer(512, 0);
    ProfilingRelationshipType relationshipType = static_cast<ProfilingRelationshipType>(5);
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    const uint64_t attributeGuid = 876345u;

    unsigned int numberOfBytesWritten = 789u;

    CHECK_THROWS_AS(WriteTimelineRelationshipBinary(relationshipType,
                                                      relationshipGuid,
                                                      headGuid,
                                                      tailGuid,
                                                      attributeGuid,
                                                      buffer.data(),
                                                      arm::pipe::numeric_cast<unsigned int>(buffer.size()),
                                                      numberOfBytesWritten),
                    arm::pipe::InvalidArgumentException);

    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineRelationshipPacketTestDataConstruction")
{
    std::vector<unsigned char> buffer(512, 0);

    ProfilingRelationshipType relationshipType = ProfilingRelationshipType::RetentionLink;
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    const uint64_t attributeGuid = 876345u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result =
                             WriteTimelineRelationshipBinary(relationshipType,
                                                             relationshipGuid,
                                                             headGuid,
                                                             tailGuid,
                                                             attributeGuid,
                                                             buffer.data(),
                                                             arm::pipe::numeric_cast<unsigned int>(buffer.size()),
                                                             numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::Ok);
    CHECK(numberOfBytesWritten == 40);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the packet header
    unsigned int offset = 0;
    // Check the decl_id
    uint32_t readDeclId = ReadUint32(buffer.data(), offset);
    CHECK(readDeclId == 3);

    // Check the relationship type
    offset += uint32_t_size;
    uint32_t readRelationshipType = ReadUint32(buffer.data(), offset);
    CHECK(readRelationshipType == 0);

    // Check the relationship GUID
    offset += uint32_t_size;
    uint64_t readRelationshipGuid = ReadUint64(buffer.data(), offset);
    CHECK(readRelationshipGuid == relationshipGuid);

    // Check the head GUID
    offset += uint64_t_size;
    uint64_t readHeadGuid = ReadUint64(buffer.data(), offset);
    CHECK(readHeadGuid == headGuid);

    // Check the tail GUID
    offset += uint64_t_size;
    uint64_t readTailGuid = ReadUint64(buffer.data(), offset);
    CHECK(readTailGuid == tailGuid);

    // Check the attribute GUID
    offset += uint64_t_size;
    uint64_t readAttributeGuid = ReadUint64(buffer.data(), offset);
    CHECK(readAttributeGuid == attributeGuid);
}

TEST_CASE("TimelineRelationshipPacketExecutionLinkTestDataConstruction")
{
    std::vector<unsigned char> buffer(512, 0);

    ProfilingRelationshipType relationshipType = ProfilingRelationshipType::ExecutionLink;
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    const uint64_t attributeGuid = 876345u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result =
                             WriteTimelineRelationshipBinary(relationshipType,
                                                             relationshipGuid,
                                                             headGuid,
                                                             tailGuid,
                                                             attributeGuid,
                                                             buffer.data(),
                                                             arm::pipe::numeric_cast<unsigned int>(buffer.size()),
                                                             numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::Ok);
    CHECK(numberOfBytesWritten == 40);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    unsigned int offset = 0;
    uint32_t readDeclId = ReadUint32(buffer.data(), offset);
    CHECK(readDeclId == 3);

    // Check the relationship type
    offset += uint32_t_size;
    uint32_t readRelationshipType = ReadUint32(buffer.data(), offset);
    CHECK(readRelationshipType == 1);

    // Check the relationship GUID
    offset += uint32_t_size;
    uint64_t readRelationshipGuid = ReadUint64(buffer.data(), offset);
    CHECK(readRelationshipGuid == relationshipGuid);

    // Check the head GUID
    offset += uint64_t_size;
    uint64_t readHeadGuid = ReadUint64(buffer.data(), offset);
    CHECK(readHeadGuid == headGuid);

    // Check the tail GUID
    offset += uint64_t_size;
    uint64_t readTailGuid = ReadUint64(buffer.data(), offset);
    CHECK(readTailGuid == tailGuid);

    // Check the attribute GUID
    offset += uint64_t_size;
    uint64_t readAttributeGuid = ReadUint64(buffer.data(), offset);
    CHECK(readAttributeGuid == attributeGuid);
}


TEST_CASE("TimelineRelationshipPacketDataLinkTestDataConstruction")
{
    std::vector<unsigned char> buffer(512, 0);

    ProfilingRelationshipType relationshipType = ProfilingRelationshipType::DataLink;
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    const uint64_t attributeGuid = 876345u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result =
                             WriteTimelineRelationshipBinary(relationshipType,
                                                             relationshipGuid,
                                                             headGuid,
                                                             tailGuid,
                                                             attributeGuid,
                                                             buffer.data(),
                                                             arm::pipe::numeric_cast<unsigned int>(buffer.size()),
                                                             numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::Ok);
    CHECK(numberOfBytesWritten == 40);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    unsigned int offset = 0;
    uint32_t readDeclId = ReadUint32(buffer.data(), offset);
    CHECK(readDeclId == 3);

    // Check the relationship type
    offset += uint32_t_size;
    uint32_t readRelationshipType = ReadUint32(buffer.data(), offset);
    CHECK(readRelationshipType == 2);

    // Check the relationship GUID
    offset += uint32_t_size;
    uint64_t readRelationshipGuid = ReadUint64(buffer.data(), offset);
    CHECK(readRelationshipGuid == relationshipGuid);

    // Check the head GUID
    offset += uint64_t_size;
    uint64_t readHeadGuid = ReadUint64(buffer.data(), offset);
    CHECK(readHeadGuid == headGuid);

    // Check the tail GUID
    offset += uint64_t_size;
    uint64_t readTailGuid = ReadUint64(buffer.data(), offset);
    CHECK(readTailGuid == tailGuid);

    // Check the attribute GUID
    offset += uint64_t_size;
    uint64_t readAttributeGuid = ReadUint64(buffer.data(), offset);
    CHECK(readAttributeGuid == attributeGuid);
}


TEST_CASE("TimelineRelationshipPacketLabelLinkTestDataConstruction")
{
    std::vector<unsigned char> buffer(512, 0);

    ProfilingRelationshipType relationshipType = ProfilingRelationshipType::LabelLink;
    const uint64_t relationshipGuid = 123456u;
    const uint64_t headGuid = 234567u;
    const uint64_t tailGuid = 345678u;
    const uint64_t attributeGuid = 876345u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result =
                             WriteTimelineRelationshipBinary(relationshipType,
                                                             relationshipGuid,
                                                             headGuid,
                                                             tailGuid,
                                                             attributeGuid,
                                                             buffer.data(),
                                                             arm::pipe::numeric_cast<unsigned int>(buffer.size()),
                                                             numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::Ok);
    CHECK(numberOfBytesWritten == 40);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the packet header
    unsigned int offset = 0;
    uint32_t readDeclId = ReadUint32(buffer.data(), offset);
    CHECK(readDeclId == 3);

    // Check the relationship type
    offset += uint32_t_size;
    uint32_t readRelationshipType = ReadUint32(buffer.data(), offset);
    CHECK(readRelationshipType == 3);

    // Check the relationship GUID
    offset += uint32_t_size;
    uint64_t readRelationshipGuid = ReadUint64(buffer.data(), offset);
    CHECK(readRelationshipGuid == relationshipGuid);

    // Check the head GUID
    offset += uint64_t_size;
    uint64_t readHeadGuid = ReadUint64(buffer.data(), offset);
    CHECK(readHeadGuid == headGuid);

    // Check the tail GUID
    offset += uint64_t_size;
    uint64_t readTailGuid = ReadUint64(buffer.data(), offset);
    CHECK(readTailGuid == tailGuid);

    // Check the attribute GUID
    offset += uint64_t_size;
    uint64_t readAttributeGuid = ReadUint64(buffer.data(), offset);
    CHECK(readAttributeGuid == attributeGuid);
}

TEST_CASE("TimelineMessageDirectoryPacketTestNoBuffer")
{
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineMessageDirectoryPackage(nullptr,
                                                                       512u,
                                                                       numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineMessageDirectoryPacketTestBufferExhausted")
{
    std::vector<unsigned char> buffer(512, 0);

    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineMessageDirectoryPackage(buffer.data(),
                                                                       0,
                                                                       numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineMessageDirectoryPacketTestFullConstruction")
{
    std::vector<unsigned char> buffer(512, 0);
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineMessageDirectoryPackage(buffer.data(),
                                                                       arm::pipe::numeric_cast<unsigned int>(
                                                                           buffer.size()),
                                                                       numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::Ok);

    CHECK(numberOfBytesWritten == 451);

    unsigned int uint8_t_size  = sizeof(uint8_t);
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the packet header
    unsigned int offset = 0;
    uint32_t packetHeaderWord0 = ReadUint32(buffer.data(), offset);
    uint32_t packetFamily = (packetHeaderWord0 >> 26) & 0x0000003F;
    uint32_t packetClass  = (packetHeaderWord0 >> 19) & 0x0000007F;
    uint32_t packetType   = (packetHeaderWord0 >> 16) & 0x00000007;
    uint32_t streamId     = (packetHeaderWord0 >>  0) & 0x00000007;
    CHECK(packetFamily == 1);
    CHECK(packetClass  == 0);
    CHECK(packetType   == 0);
    CHECK(streamId     == 0);

    offset += uint32_t_size;
    uint32_t packetHeaderWord1 = ReadUint32(buffer.data(), offset);
    uint32_t sequenceNumbered = (packetHeaderWord1 >> 24) & 0x00000001;
    uint32_t dataLength       = (packetHeaderWord1 >>  0) & 0x00FFFFFF;
    CHECK(sequenceNumbered ==  0);
    CHECK(dataLength       == 443);

    // Check the stream header
    offset += uint32_t_size;
    uint8_t readStreamVersion = ReadUint8(buffer.data(), offset);
    CHECK(readStreamVersion == 4);
    offset += uint8_t_size;
    uint8_t readPointerBytes = ReadUint8(buffer.data(), offset);
    CHECK(readPointerBytes == uint64_t_size);
    offset += uint8_t_size;
    uint8_t readThreadIdBytes = ReadUint8(buffer.data(), offset);
    CHECK(readThreadIdBytes == ThreadIdSize);

    // Check the number of declarations
    offset += uint8_t_size;
    uint32_t declCount = ReadUint32(buffer.data(), offset);
    CHECK(declCount == 5);

    // Check the decl_id
    offset += uint32_t_size;
    uint32_t readDeclId = ReadUint32(buffer.data(), offset);
    CHECK(readDeclId == 0);

    // SWTrace "namestring" format
    // length of the string (first 4 bytes) + string + null terminator

    // Check the decl_name
    offset += uint32_t_size;
    uint32_t swTraceDeclNameLength = ReadUint32(buffer.data(), offset);
    CHECK(swTraceDeclNameLength == 13); // decl_name length including the null-terminator

    std::string label = "declareLabel";
    offset += uint32_t_size;
    CHECK(std::memcmp(buffer.data() + offset,           // Offset to the label in the buffer
                            label.data(),                     // The original label
                            swTraceDeclNameLength - 1) == 0); // The length of the label

    // Check the ui_name
    std::vector<uint32_t> swTraceString;
    StringToSwTraceString<SwTraceCharPolicy>(label, swTraceString);
    offset += (arm::pipe::numeric_cast<unsigned int>(swTraceString.size()) - 1) * uint32_t_size;
    uint32_t swTraceUINameLength = ReadUint32(buffer.data(), offset);
    CHECK(swTraceUINameLength == 14); // ui_name length including the null-terminator

    label = "declare label";
    offset += uint32_t_size;
    CHECK(std::memcmp(buffer.data() + offset,           // Offset to the label in the buffer
                            label.data(),                     // The original label
                            swTraceUINameLength - 1) == 0);   // The length of the label

    // Check arg_types
    StringToSwTraceString<SwTraceCharPolicy>(label, swTraceString);
    offset += (arm::pipe::numeric_cast<unsigned int>(swTraceString.size()) - 1) * uint32_t_size;
    uint32_t swTraceArgTypesLength = ReadUint32(buffer.data(), offset);
    CHECK(swTraceArgTypesLength == 3); // arg_types length including the null-terminator

    label = "ps";
    offset += uint32_t_size;
    CHECK(std::memcmp(buffer.data() + offset,           // Offset to the label in the buffer
                            label.data(),                     // The original label
                            swTraceArgTypesLength - 1) == 0); // The length of the label

    // Check arg_names
    StringToSwTraceString<SwTraceCharPolicy>(label, swTraceString);
    offset += (arm::pipe::numeric_cast<unsigned int>(swTraceString.size()) - 1) * uint32_t_size;
    uint32_t swTraceArgNamesLength = ReadUint32(buffer.data(), offset);
    CHECK(swTraceArgNamesLength == 11); // arg_names length including the null-terminator

    label = "guid,value";
    offset += uint32_t_size;
    CHECK(std::memcmp(buffer.data() + offset,        // Offset to the label in the buffer
                            label.data(),                     // The original label
                            swTraceArgNamesLength - 1) == 0); // The length of the label

    // Check second message decl_id
    StringToSwTraceString<SwTraceCharPolicy>(label, swTraceString);
    offset += (arm::pipe::numeric_cast<unsigned int>(swTraceString.size()) - 1) * uint32_t_size;
    readDeclId = ReadUint32(buffer.data(), offset);
    CHECK(readDeclId == 1);

    // Check second decl_name
    offset += uint32_t_size;
    swTraceDeclNameLength = ReadUint32(buffer.data(), offset);
    CHECK(swTraceDeclNameLength == 14); // decl_name length including the null-terminator

    label = "declareEntity";
    offset += uint32_t_size;
    CHECK(std::memcmp(buffer.data() + offset,           // Offset to the label in the buffer
                            label.data(),                     // The original label
                            swTraceDeclNameLength - 1) == 0); // The length of the label
}

TEST_CASE("TimelineEntityPacketTestNoBuffer")
{
    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEntityBinary(profilingGuid,
                                                            nullptr,
                                                            512u,
                                                            numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineEntityPacketTestBufferExhaustedWithZeroBufferSize")
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEntityBinary(profilingGuid,
                                                            buffer.data(),
                                                            0,
                                                            numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineEntityPacketTestBufferExhaustedWithFixedBufferSize")
{
    std::vector<unsigned char> buffer(10, 0);

    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEntityBinary(profilingGuid,
                                                            buffer.data(),
                                                            arm::pipe::numeric_cast<unsigned int>(buffer.size()),
                                                            numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineEntityPacketTestFullConstructionOfData")
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEntityBinary(profilingGuid,
                                                            buffer.data(),
                                                            arm::pipe::numeric_cast<unsigned int>(buffer.size()),
                                                            numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::Ok);
    CHECK(numberOfBytesWritten == 12);

    unsigned int uint32_t_size = sizeof(uint32_t);

    unsigned int offset = 0;
    // Check decl_Id
    uint32_t decl_Id = ReadUint32(buffer.data(), offset);
    CHECK(decl_Id == uint32_t(1));

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(buffer.data(), offset);
    CHECK(readProfilingGuid == profilingGuid);
}

TEST_CASE("TimelineEventClassTestNoBuffer")
{
    const uint64_t profilingGuid = 123456u;
    const uint64_t profilingNameGuid = 3345u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventClassBinary(profilingGuid,
                                                                profilingNameGuid,
                                                                nullptr,
                                                                512u,
                                                                numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineEventClassTestBufferExhaustionZeroValue")
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    const uint64_t profilingNameGuid = 3345u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventClassBinary(profilingGuid,
                                                                profilingNameGuid,
                                                                buffer.data(),
                                                                0,
                                                                numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineEventClassTestBufferExhaustionFixedValue")
{
    std::vector<unsigned char> buffer(10, 0);

    const uint64_t profilingGuid = 123456u;
    const uint64_t profilingNameGuid = 5564u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventClassBinary(profilingGuid,
                                                                profilingNameGuid,
                                                                buffer.data(),
                                                                arm::pipe::numeric_cast<unsigned int>(buffer.size()),
                                                                numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineEventClassTestFullConstructionOfData")
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t profilingGuid = 123456u;
    const uint64_t profilingNameGuid = 654321u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventClassBinary(profilingGuid,
                                                                profilingNameGuid,
                                                                buffer.data(),
                                                                arm::pipe::numeric_cast<unsigned int>(buffer.size()),
                                                                numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::Ok);
    CHECK(numberOfBytesWritten == 20);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    unsigned int offset = 0;
    // Check the decl_id
    uint32_t declId = ReadUint32(buffer.data(), offset);
    CHECK(declId == uint32_t(2));

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(buffer.data(), offset);
    CHECK(readProfilingGuid == profilingGuid);

    offset += uint64_t_size;
    uint64_t readProfilingNameGuid = ReadUint64(buffer.data(), offset);
    CHECK(readProfilingNameGuid == profilingNameGuid);
}

TEST_CASE("TimelineEventPacketTestNoBuffer")
{
    const uint64_t timestamp = 456789u;
    const int threadId = arm::pipe::GetCurrentThreadId();
    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventBinary(timestamp,
                                                           threadId,
                                                           profilingGuid,
                                                           nullptr,
                                                           512u,
                                                           numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineEventPacketTestBufferExhaustionZeroValue")
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t timestamp = 456789u;
    const int threadId = arm::pipe::GetCurrentThreadId();
    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventBinary(timestamp,
                                                           threadId,
                                                           profilingGuid,
                                                           buffer.data(),
                                                           0,
                                                           numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineEventPacketTestBufferExhaustionFixedValue")
{
    std::vector<unsigned char> buffer(10, 0);

    const uint64_t timestamp = 456789u;
    const int threadId = arm::pipe::GetCurrentThreadId();
    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventBinary(timestamp,
                                                           threadId,
                                                           profilingGuid,
                                                           buffer.data(),
                                                           arm::pipe::numeric_cast<unsigned int>(buffer.size()),
                                                           numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::BufferExhaustion);
    CHECK(numberOfBytesWritten == 0);
}

TEST_CASE("TimelineEventPacketTestFullConstructionOfData")
{
    std::vector<unsigned char> buffer(512, 0);

    const uint64_t timestamp = 456789u;
    const int threadId = arm::pipe::GetCurrentThreadId();
    const uint64_t profilingGuid = 123456u;
    unsigned int numberOfBytesWritten = 789u;
    TimelinePacketStatus result = WriteTimelineEventBinary(timestamp,
                                                           threadId,
                                                           profilingGuid,
                                                           buffer.data(),
                                                           arm::pipe::numeric_cast<unsigned int>(buffer.size()),
                                                           numberOfBytesWritten);
    CHECK(result == TimelinePacketStatus::Ok);

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);
    CHECK(numberOfBytesWritten == 20 + ThreadIdSize);

    unsigned int offset = 0;
    // Check the decl_id
    uint32_t readDeclId = ReadUint32(buffer.data(), offset);
    CHECK(readDeclId == 4);

    // Check the timestamp
    offset += uint32_t_size;
    uint64_t readTimestamp = ReadUint64(buffer.data(), offset);
    CHECK(readTimestamp == timestamp);

    // Check the thread id
    offset += uint64_t_size;
    std::vector<uint8_t> readThreadId(ThreadIdSize, 0);
    ReadBytes(buffer.data(), offset, ThreadIdSize, readThreadId.data());
    CHECK(readThreadId == threadId);

    // Check the profiling GUID
    offset += ThreadIdSize;
    uint64_t readProfilingGuid = ReadUint64(buffer.data(), offset);
    CHECK(readProfilingGuid == profilingGuid);
}

}
