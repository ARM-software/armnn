//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingMocks.hpp"

#include <BufferManager.hpp>
#include <ProfilingService.hpp>
#include <ProfilingUtils.hpp>
#include <SendTimelinePacket.hpp>
#include <armnnUtils/Threads.hpp>
#include <TimelinePacketWriterFactory.hpp>

#include <common/include/SwTrace.hpp>
#include <common/include/LabelsAndEventClasses.hpp>

#include <doctest/doctest.h>

#include <functional>
#include <Runtime.hpp>

using namespace armnn::profiling;

TEST_SUITE("SendTimelinePacketTests")
{
TEST_CASE("SendTimelineMessageDirectoryPackageTest")
{
    MockBufferManager mockBuffer(512);
    TimelinePacketWriterFactory timelinePacketWriterFactory(mockBuffer);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = timelinePacketWriterFactory.GetSendTimelinePacket();

    sendTimelinePacket->SendTimelineMessageDirectoryPackage();

    // Get the readable buffer
    auto packetBuffer = mockBuffer.GetReadableBuffer();

    unsigned int uint8_t_size  = sizeof(uint8_t);
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the packet header
    unsigned int offset = 0;
    uint32_t packetHeaderWord0 = ReadUint32(packetBuffer, offset);
    uint32_t packetFamily = (packetHeaderWord0 >> 26) & 0x0000003F;
    uint32_t packetClass  = (packetHeaderWord0 >> 19) & 0x0000007F;
    uint32_t packetType   = (packetHeaderWord0 >> 16) & 0x00000007;
    uint32_t streamId     = (packetHeaderWord0 >>  0) & 0x00000007;

    CHECK(packetFamily == 1);
    CHECK(packetClass  == 0);
    CHECK(packetType   == 0);
    CHECK(streamId     == 0);

    offset += uint32_t_size;
    uint32_t packetHeaderWord1 = ReadUint32(packetBuffer, offset);
    uint32_t sequenceNumbered = (packetHeaderWord1 >> 24) & 0x00000001;
    uint32_t dataLength       = (packetHeaderWord1 >>  0) & 0x00FFFFFF;
    CHECK(sequenceNumbered ==  0);
    CHECK(dataLength       == 443);

    offset += uint32_t_size;
    uint8_t readStreamVersion = ReadUint8(packetBuffer, offset);
    CHECK(readStreamVersion == 4);
    offset += uint8_t_size;
    uint8_t readPointerBytes = ReadUint8(packetBuffer, offset);
    CHECK(readPointerBytes == uint64_t_size);
    offset += uint8_t_size;
    uint8_t readThreadIdBytes = ReadUint8(packetBuffer, offset);
    CHECK(readThreadIdBytes == ThreadIdSize);

    offset += uint8_t_size;
    uint32_t DeclCount = ReadUint32(packetBuffer, offset);
    CHECK(DeclCount == 5);

    offset += uint32_t_size;
    arm::pipe::SwTraceMessage swTraceMessage = arm::pipe::ReadSwTraceMessage(packetBuffer->GetReadableData(),
                                                                             offset,
                                                                             packetBuffer->GetSize());

    CHECK(swTraceMessage.m_Id == 0);
    CHECK(swTraceMessage.m_Name == "declareLabel");
    CHECK(swTraceMessage.m_UiName == "declare label");
    CHECK(swTraceMessage.m_ArgTypes.size() == 2);
    CHECK(swTraceMessage.m_ArgTypes[0] == 'p');
    CHECK(swTraceMessage.m_ArgTypes[1] == 's');
    CHECK(swTraceMessage.m_ArgNames.size() == 2);
    CHECK(swTraceMessage.m_ArgNames[0] == "guid");
    CHECK(swTraceMessage.m_ArgNames[1] == "value");

    swTraceMessage = arm::pipe::ReadSwTraceMessage(packetBuffer->GetReadableData(),
                                                   offset,
                                                   packetBuffer->GetSize());

    CHECK(swTraceMessage.m_Id == 1);
    CHECK(swTraceMessage.m_Name == "declareEntity");
    CHECK(swTraceMessage.m_UiName == "declare entity");
    CHECK(swTraceMessage.m_ArgTypes.size() == 1);
    CHECK(swTraceMessage.m_ArgTypes[0] == 'p');
    CHECK(swTraceMessage.m_ArgNames.size() == 1);
    CHECK(swTraceMessage.m_ArgNames[0] == "guid");

    swTraceMessage = arm::pipe::ReadSwTraceMessage(packetBuffer->GetReadableData(),
                                                   offset,
                                                   packetBuffer->GetSize());

    CHECK(swTraceMessage.m_Id == 2);
    CHECK(swTraceMessage.m_Name == "declareEventClass");
    CHECK(swTraceMessage.m_UiName == "declare event class");
    CHECK(swTraceMessage.m_ArgTypes.size() == 2);
    CHECK(swTraceMessage.m_ArgTypes[0] == 'p');
    CHECK(swTraceMessage.m_ArgTypes[1] == 'p');
    CHECK(swTraceMessage.m_ArgNames.size() == 2);
    CHECK(swTraceMessage.m_ArgNames[0] == "guid");
    CHECK(swTraceMessage.m_ArgNames[1] == "nameGuid");

    swTraceMessage = arm::pipe::ReadSwTraceMessage(packetBuffer->GetReadableData(),
                                                   offset,
                                                   packetBuffer->GetSize());

    CHECK(swTraceMessage.m_Id == 3);
    CHECK(swTraceMessage.m_Name == "declareRelationship");
    CHECK(swTraceMessage.m_UiName == "declare relationship");
    CHECK(swTraceMessage.m_ArgTypes.size() == 5);
    CHECK(swTraceMessage.m_ArgTypes[0] == 'I');
    CHECK(swTraceMessage.m_ArgTypes[1] == 'p');
    CHECK(swTraceMessage.m_ArgTypes[2] == 'p');
    CHECK(swTraceMessage.m_ArgTypes[3] == 'p');
    CHECK(swTraceMessage.m_ArgTypes[4] == 'p');
    CHECK(swTraceMessage.m_ArgNames.size() == 5);
    CHECK(swTraceMessage.m_ArgNames[0] == "relationshipType");
    CHECK(swTraceMessage.m_ArgNames[1] == "relationshipGuid");
    CHECK(swTraceMessage.m_ArgNames[2] == "headGuid");
    CHECK(swTraceMessage.m_ArgNames[3] == "tailGuid");
    CHECK(swTraceMessage.m_ArgNames[4] == "attributeGuid");

    swTraceMessage = arm::pipe::ReadSwTraceMessage(packetBuffer->GetReadableData(),
                                                   offset,
                                                   packetBuffer->GetSize());

    CHECK(swTraceMessage.m_Id == 4);
    CHECK(swTraceMessage.m_Name == "declareEvent");
    CHECK(swTraceMessage.m_UiName == "declare event");
    CHECK(swTraceMessage.m_ArgTypes.size() == 3);
    CHECK(swTraceMessage.m_ArgTypes[0] == '@');
    CHECK(swTraceMessage.m_ArgTypes[1] == 't');
    CHECK(swTraceMessage.m_ArgTypes[2] == 'p');
    CHECK(swTraceMessage.m_ArgNames.size() == 3);
    CHECK(swTraceMessage.m_ArgNames[0] == "timestamp");
    CHECK(swTraceMessage.m_ArgNames[1] == "threadId");
    CHECK(swTraceMessage.m_ArgNames[2] == "eventGuid");
}

TEST_CASE("SendTimelineEntityWithEventClassPacketTest")
{
    MockBufferManager bufferManager(40);
    TimelinePacketWriterFactory timelinePacketWriterFactory(bufferManager);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = timelinePacketWriterFactory.GetSendTimelinePacket();

    const uint64_t entityBinaryPacketProfilingGuid = 123456u;
    sendTimelinePacket->SendTimelineEntityBinaryPacket(entityBinaryPacketProfilingGuid);

    const uint64_t eventClassBinaryPacketProfilingGuid = 789123u;
    const uint64_t eventClassBinaryPacketNameGuid = 8845u;
    sendTimelinePacket->SendTimelineEventClassBinaryPacket(
        eventClassBinaryPacketProfilingGuid, eventClassBinaryPacketNameGuid);

    // Commit the messages
    sendTimelinePacket->Commit();

    // Get the readable buffer
    auto packetBuffer = bufferManager.GetReadableBuffer();

    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    // Check the packet header
    unsigned int offset = 0;

    // Reading TimelineEntityClassBinaryPacket
    uint32_t entityBinaryPacketHeaderWord0  = ReadUint32(packetBuffer, offset);
    uint32_t entityBinaryPacketFamily       = (entityBinaryPacketHeaderWord0 >> 26) & 0x0000003F;
    uint32_t entityBinaryPacketClass        = (entityBinaryPacketHeaderWord0 >> 19) & 0x0000007F;
    uint32_t entityBinaryPacketType         = (entityBinaryPacketHeaderWord0 >> 16) & 0x00000007;
    uint32_t entityBinaryPacketStreamId     = (entityBinaryPacketHeaderWord0 >>  0) & 0x00000007;

    CHECK(entityBinaryPacketFamily       == 1);
    CHECK(entityBinaryPacketClass        == 0);
    CHECK(entityBinaryPacketType         == 1);
    CHECK(entityBinaryPacketStreamId     == 0);

    offset += uint32_t_size;

    uint32_t entityBinaryPacketHeaderWord1      = ReadUint32(packetBuffer, offset);

    uint32_t entityBinaryPacketSequenceNumbered = (entityBinaryPacketHeaderWord1 >> 24) & 0x00000001;
    uint32_t entityBinaryPacketDataLength       = (entityBinaryPacketHeaderWord1 >>  0) & 0x00FFFFFF;

    CHECK(entityBinaryPacketSequenceNumbered == 0);
    CHECK(entityBinaryPacketDataLength       == 32);

    // Check the decl_id
    offset += uint32_t_size;
    uint32_t entitytDecId = ReadUint32(packetBuffer, offset);

    CHECK(entitytDecId == uint32_t(1));

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(packetBuffer, offset);

    CHECK(readProfilingGuid == entityBinaryPacketProfilingGuid);

    // Reading TimelineEventClassBinaryPacket
    offset += uint64_t_size;

    uint32_t eventClassDeclId = ReadUint32(packetBuffer, offset);
    CHECK(eventClassDeclId == uint32_t(2));

    // Check the profiling GUID
    offset += uint32_t_size;
    readProfilingGuid = ReadUint64(packetBuffer, offset);
    CHECK(readProfilingGuid == eventClassBinaryPacketProfilingGuid);

    offset += uint64_t_size;
    uint64_t readEventClassNameGuid = ReadUint64(packetBuffer, offset);
    CHECK(readEventClassNameGuid == eventClassBinaryPacketNameGuid);

    bufferManager.MarkRead(packetBuffer);
}

TEST_CASE("SendEventClassAfterTimelineEntityPacketTest")
{
    unsigned int uint32_t_size = sizeof(uint32_t);
    unsigned int uint64_t_size = sizeof(uint64_t);

    MockBufferManager bufferManager(512);
    TimelinePacketWriterFactory timelinePacketWriterFactory(bufferManager);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = timelinePacketWriterFactory.GetSendTimelinePacket();

    // Send TimelineEntityClassBinaryPacket
    const uint64_t entityBinaryPacketProfilingGuid = 123456u;
    sendTimelinePacket->SendTimelineEntityBinaryPacket(entityBinaryPacketProfilingGuid);

    // Commit the buffer
    sendTimelinePacket->Commit();

    // Get the readable buffer
    auto packetBuffer = bufferManager.GetReadableBuffer();

    // Check the packet header
    unsigned int offset = 0;

    // Reading TimelineEntityClassBinaryPacket
    uint32_t entityBinaryPacketHeaderWord0 = ReadUint32(packetBuffer, offset);
    uint32_t entityBinaryPacketFamily = (entityBinaryPacketHeaderWord0 >> 26) & 0x0000003F;
    uint32_t entityBinaryPacketClass  = (entityBinaryPacketHeaderWord0 >> 19) & 0x0000007F;
    uint32_t entityBinaryPacketType   = (entityBinaryPacketHeaderWord0 >> 16) & 0x00000007;
    uint32_t entityBinaryPacketStreamId     = (entityBinaryPacketHeaderWord0 >>  0) & 0x00000007;

    CHECK(entityBinaryPacketFamily == 1);
    CHECK(entityBinaryPacketClass  == 0);
    CHECK(entityBinaryPacketType   == 1);
    CHECK(entityBinaryPacketStreamId     == 0);

    offset += uint32_t_size;
    uint32_t entityBinaryPacketHeaderWord1 = ReadUint32(packetBuffer, offset);
    uint32_t entityBinaryPacketSequenceNumbered = (entityBinaryPacketHeaderWord1 >> 24) & 0x00000001;
    uint32_t entityBinaryPacketDataLength       = (entityBinaryPacketHeaderWord1 >>  0) & 0x00FFFFFF;
    CHECK(entityBinaryPacketSequenceNumbered == 0);
    CHECK(entityBinaryPacketDataLength       == 12);

    // Check the decl_id
    offset += uint32_t_size;
    uint32_t entitytDecId = ReadUint32(packetBuffer, offset);

    CHECK(entitytDecId == uint32_t(1));

    // Check the profiling GUID
    offset += uint32_t_size;
    uint64_t readProfilingGuid = ReadUint64(packetBuffer, offset);

    CHECK(readProfilingGuid == entityBinaryPacketProfilingGuid);

    bufferManager.MarkRead(packetBuffer);

    // Send TimelineEventClassBinaryPacket
    const uint64_t eventClassBinaryPacketProfilingGuid = 789123u;
    const uint64_t eventClassBinaryPacketNameGuid = 8845u;
    sendTimelinePacket->SendTimelineEventClassBinaryPacket(
        eventClassBinaryPacketProfilingGuid, eventClassBinaryPacketNameGuid);

    // Commit the buffer
    sendTimelinePacket->Commit();

    // Get the readable buffer
    packetBuffer = bufferManager.GetReadableBuffer();

    // Check the packet header
    offset = 0;

    // Reading TimelineEventClassBinaryPacket
    uint32_t eventClassBinaryPacketHeaderWord0 = ReadUint32(packetBuffer, offset);
    uint32_t eventClassBinaryPacketFamily = (eventClassBinaryPacketHeaderWord0 >> 26) & 0x0000003F;
    uint32_t eventClassBinaryPacketClass  = (eventClassBinaryPacketHeaderWord0 >> 19) & 0x0000007F;
    uint32_t eventClassBinaryPacketType   = (eventClassBinaryPacketHeaderWord0 >> 16) & 0x00000007;
    uint32_t eventClassBinaryPacketStreamId     = (eventClassBinaryPacketHeaderWord0 >>  0) & 0x00000007;

    CHECK(eventClassBinaryPacketFamily == 1);
    CHECK(eventClassBinaryPacketClass  == 0);
    CHECK(eventClassBinaryPacketType   == 1);
    CHECK(eventClassBinaryPacketStreamId     == 0);

    offset += uint32_t_size;
    uint32_t eventClassBinaryPacketHeaderWord1 = ReadUint32(packetBuffer, offset);
    uint32_t eventClassBinaryPacketSequenceNumbered = (eventClassBinaryPacketHeaderWord1 >> 24) & 0x00000001;
    uint32_t eventClassBinaryPacketDataLength       = (eventClassBinaryPacketHeaderWord1 >>  0) & 0x00FFFFFF;
    CHECK(eventClassBinaryPacketSequenceNumbered == 0);
    CHECK(eventClassBinaryPacketDataLength       == 20);

    offset += uint32_t_size;
    uint32_t eventClassDeclId = ReadUint32(packetBuffer, offset);
    CHECK(eventClassDeclId == uint32_t(2));

    // Check the profiling GUID
    offset += uint32_t_size;
    readProfilingGuid = ReadUint64(packetBuffer, offset);
    CHECK(readProfilingGuid == eventClassBinaryPacketProfilingGuid);

    offset += uint64_t_size;
    uint64_t readEventClassNameGuid = ReadUint64(packetBuffer, offset);
    CHECK(readEventClassNameGuid == eventClassBinaryPacketNameGuid);

    bufferManager.MarkRead(packetBuffer);

    // Send TimelineEventBinaryPacket
    const uint64_t timestamp = 456789u;
    const int threadId = armnnUtils::Threads::GetCurrentThreadId();
    const uint64_t eventProfilingGuid = 123456u;
    sendTimelinePacket->SendTimelineEventBinaryPacket(timestamp, threadId, eventProfilingGuid);

    // Commit the buffer
    sendTimelinePacket->Commit();

    // Get the readable buffer
    packetBuffer = bufferManager.GetReadableBuffer();

    // Check the packet header
    offset = 0;

    // Reading TimelineEventBinaryPacket
    uint32_t eventBinaryPacketHeaderWord0 = ReadUint32(packetBuffer, offset);
    uint32_t eventBinaryPacketFamily = (eventBinaryPacketHeaderWord0 >> 26) & 0x0000003F;
    uint32_t eventBinaryPacketClass  = (eventBinaryPacketHeaderWord0 >> 19) & 0x0000007F;
    uint32_t eventBinaryPacketType   = (eventBinaryPacketHeaderWord0 >> 16) & 0x00000007;
    uint32_t eventBinaryPacketStreamId     = (eventBinaryPacketHeaderWord0 >>  0) & 0x00000007;

    CHECK(eventBinaryPacketFamily == 1);
    CHECK(eventBinaryPacketClass  == 0);
    CHECK(eventBinaryPacketType   == 1);
    CHECK(eventBinaryPacketStreamId     == 0);

    offset += uint32_t_size;
    uint32_t eventBinaryPacketHeaderWord1 = ReadUint32(packetBuffer, offset);
    uint32_t eventBinaryPacketSequenceNumbered = (eventBinaryPacketHeaderWord1 >> 24) & 0x00000001;
    uint32_t eventBinaryPacketDataLength       = (eventBinaryPacketHeaderWord1 >>  0) & 0x00FFFFFF;
    CHECK(eventBinaryPacketSequenceNumbered == 0);
    CHECK(eventBinaryPacketDataLength == 20 + ThreadIdSize);

    // Check the decl_id
    offset += uint32_t_size;
    uint32_t eventDeclId = ReadUint32(packetBuffer, offset);
    CHECK(eventDeclId == 4);

    // Check the timestamp
    offset += uint32_t_size;
    uint64_t eventTimestamp = ReadUint64(packetBuffer, offset);
    CHECK(eventTimestamp == timestamp);

    // Check the thread id
    offset += uint64_t_size;
    std::vector<uint8_t> readThreadId(ThreadIdSize, 0);
    ReadBytes(packetBuffer, offset, ThreadIdSize, readThreadId.data());
    CHECK(readThreadId == threadId);

    // Check the profiling GUID
    offset += ThreadIdSize;
    readProfilingGuid = ReadUint64(packetBuffer, offset);
    CHECK(readProfilingGuid == eventProfilingGuid);
}

TEST_CASE("SendTimelinePacketTests2")
{
    MockBufferManager bufferManager(40);
    TimelinePacketWriterFactory timelinePacketWriterFactory(bufferManager);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = timelinePacketWriterFactory.GetSendTimelinePacket();

    CHECK_THROWS_AS(sendTimelinePacket->SendTimelineMessageDirectoryPackage(),
                      armnn::RuntimeException);
}

TEST_CASE("SendTimelinePacketTests3")
{
    MockBufferManager bufferManager(512);
    TimelinePacketWriterFactory timelinePacketWriterFactory(bufferManager);
    std::unique_ptr<ISendTimelinePacket> sendTimelinePacket = timelinePacketWriterFactory.GetSendTimelinePacket();

    // Send TimelineEntityClassBinaryPacket
    const uint64_t entityBinaryPacketProfilingGuid = 123456u;
    sendTimelinePacket->SendTimelineEntityBinaryPacket(entityBinaryPacketProfilingGuid);

    // Commit the buffer
    sendTimelinePacket->Commit();

    // Get the readable buffer
    auto packetBuffer = bufferManager.GetReadableBuffer();

    // Send TimelineEventClassBinaryPacket
    const uint64_t eventClassBinaryPacketProfilingGuid = 789123u;
    const uint64_t eventClassBinaryPacketNameGuid = 8845u;
    CHECK_THROWS_AS(sendTimelinePacket->SendTimelineEventClassBinaryPacket(
                      eventClassBinaryPacketProfilingGuid, eventClassBinaryPacketNameGuid),
                      armnn::profiling::BufferExhaustion);
}

TEST_CASE("GetGuidsFromProfilingService")
{
    armnn::IRuntime::CreationOptions options;
    options.m_ProfilingOptions.m_EnableProfiling = true;
    armnn::RuntimeImpl runtime(options);
    armnn::profiling::ProfilingService profilingService(runtime);

    profilingService.ResetExternalProfilingOptions(options.m_ProfilingOptions, true);
    ProfilingStaticGuid staticGuid = profilingService.GetStaticId("dummy");
    std::hash<std::string> hasher;
    uint64_t hash = static_cast<uint64_t>(hasher("dummy"));
    ProfilingStaticGuid expectedStaticValue(hash | MIN_STATIC_GUID);
    CHECK(staticGuid == expectedStaticValue);
    ProfilingDynamicGuid dynamicGuid = profilingService.GetNextGuid();
    uint64_t dynamicGuidValue = static_cast<uint64_t>(dynamicGuid);
    ++dynamicGuidValue;
    ProfilingDynamicGuid expectedDynamicValue(dynamicGuidValue);
    dynamicGuid = profilingService.GetNextGuid();
    CHECK(dynamicGuid == expectedDynamicValue);
}

TEST_CASE("GetTimelinePackerWriterFromProfilingService")
{
    armnn::IRuntime::CreationOptions::ExternalProfilingOptions options;
    options.m_EnableProfiling = true;
    armnn::profiling::ProfilingService profilingService;
    profilingService.ResetExternalProfilingOptions(options, true);

    std::unique_ptr<ISendTimelinePacket> writer = profilingService.GetSendTimelinePacket();
    CHECK(writer != nullptr);
}

TEST_CASE("CheckStaticGuidsAndEvents")
{
    CHECK("name" == LabelsAndEventClasses::NAME_LABEL);
    CHECK("type" == LabelsAndEventClasses::TYPE_LABEL);
    CHECK("index" == LabelsAndEventClasses::INDEX_LABEL);

    std::hash<std::string> hasher;

    uint64_t hash = static_cast<uint64_t>(hasher(LabelsAndEventClasses::NAME_LABEL));
    ProfilingStaticGuid expectedNameGuid(hash | MIN_STATIC_GUID);
    CHECK(LabelsAndEventClasses::NAME_GUID == expectedNameGuid);

    hash = static_cast<uint64_t>(hasher(LabelsAndEventClasses::TYPE_LABEL));
    ProfilingStaticGuid expectedTypeGuid(hash | MIN_STATIC_GUID);
    CHECK(LabelsAndEventClasses::TYPE_GUID == expectedTypeGuid);

    hash = static_cast<uint64_t>(hasher(LabelsAndEventClasses::INDEX_LABEL));
    ProfilingStaticGuid expectedIndexGuid(hash | MIN_STATIC_GUID);
    CHECK(LabelsAndEventClasses::INDEX_GUID == expectedIndexGuid);

    hash = static_cast<uint64_t>(hasher("ARMNN_PROFILING_SOL"));
    ProfilingStaticGuid expectedSol(hash | MIN_STATIC_GUID);
    CHECK(LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS == expectedSol);

    hash = static_cast<uint64_t>(hasher("ARMNN_PROFILING_EOL"));
    ProfilingStaticGuid expectedEol(hash | MIN_STATIC_GUID);
    CHECK(LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS == expectedEol);
}

}
