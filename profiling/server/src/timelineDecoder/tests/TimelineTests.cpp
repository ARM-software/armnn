//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <common/include/CommandHandlerFunctor.hpp>
#include <common/include/CommonProfilingUtils.hpp>
#include <common/include/Threads.hpp>
#include <server/include/timelineDecoder/TimelineCaptureCommandHandler.hpp>
#include <server/include/timelineDecoder/TimelineDirectoryCaptureCommandHandler.hpp>
#include <server/include/timelineDecoder/TimelineDecoder.hpp>

#include <client/src/BufferManager.hpp>
#include <client/src/ProfilingService.hpp>
#include <client/src/PacketBuffer.hpp>
#include <client/src/TimelinePacketWriterFactory.hpp>

#include <doctest/doctest.h>

TEST_SUITE("TimelineDecoderTests")
{
void SendTimelinePacketToCommandHandler(const unsigned char* packetBuffer,
                                        arm::pipe::CommandHandlerFunctor& CommandHandler)
{
    uint32_t uint32_t_size = sizeof(uint32_t);
    unsigned int offset = 0;

    uint32_t header[2];
    header[0] = arm::pipe::ReadUint32(packetBuffer, offset);
    offset += uint32_t_size;
    header[1] = arm::pipe::ReadUint32(packetBuffer, offset);
    offset += uint32_t_size;
    uint32_t PacketDataLength  = header[1] & 0x00FFFFFF;

    auto uniquePacketData = std::make_unique<unsigned char[]>(PacketDataLength);
    std::memcpy(uniquePacketData.get(), packetBuffer + offset, PacketDataLength);

    arm::pipe::Packet packet(header[0], PacketDataLength, uniquePacketData);

    CHECK(std::memcmp(packetBuffer + offset, packet.GetData(), packet.GetLength()) == 0);

    CommandHandler(packet);
}

void PushEntity(arm::pipe::TimelineDecoder::Model& model, const arm::pipe::ITimelineDecoder::Entity entity)
{
    model.m_Entities.emplace_back(entity);
}

void PushEventClass(arm::pipe::TimelineDecoder::Model& model, const arm::pipe::ITimelineDecoder::EventClass eventClass)
{
    model.m_EventClasses.emplace_back(eventClass);
}

void PushEvent(arm::pipe::TimelineDecoder::Model& model, const arm::pipe::ITimelineDecoder::Event event)
{
    model.m_Events.emplace_back(event);
}

void PushLabel(arm::pipe::TimelineDecoder::Model& model, const arm::pipe::ITimelineDecoder::Label label)
{
    model.m_Labels.emplace_back(label);
}

void PushRelationship(arm::pipe::TimelineDecoder::Model& model,
                      const arm::pipe::ITimelineDecoder::Relationship relationship)
{
    model.m_Relationships.emplace_back(relationship);
}

TEST_CASE("TimelineDirectoryTest")
{
    uint32_t uint8_t_size  = sizeof(uint8_t);
    uint32_t uint32_t_size = sizeof(uint32_t);
    uint32_t uint64_t_size = sizeof(uint64_t);

    arm::pipe::BufferManager bufferManager(5);
    arm::pipe::TimelinePacketWriterFactory timelinePacketWriterFactory(bufferManager);

    std::unique_ptr<arm::pipe::ISendTimelinePacket> sendTimelinePacket =
            timelinePacketWriterFactory.GetSendTimelinePacket();

    arm::pipe::PacketVersionResolver packetVersionResolver;

    arm::pipe::TimelineDecoder timelineDecoder;
    arm::pipe::TimelineCaptureCommandHandler timelineCaptureCommandHandler(
            1, 1, packetVersionResolver.ResolvePacketVersion(1, 1).GetEncodedValue(), timelineDecoder);

    arm::pipe::TimelineDirectoryCaptureCommandHandler timelineDirectoryCaptureCommandHandler(
            1, 0, packetVersionResolver.ResolvePacketVersion(1, 0).GetEncodedValue(),
            timelineCaptureCommandHandler, true);

    sendTimelinePacket->SendTimelineMessageDirectoryPackage();
    sendTimelinePacket->Commit();

    std::vector<arm::pipe::SwTraceMessage> swTraceBufferMessages;

    unsigned int offset = uint32_t_size * 2;

    std::unique_ptr<arm::pipe::IPacketBuffer> packetBuffer = bufferManager.GetReadableBuffer();

    uint8_t readStreamVersion = ReadUint8(packetBuffer, offset);
    CHECK(readStreamVersion == 4);
    offset += uint8_t_size;
    uint8_t readPointerBytes = ReadUint8(packetBuffer, offset);
    CHECK(readPointerBytes == uint64_t_size);
    offset += uint8_t_size;
    uint8_t readThreadIdBytes = ReadUint8(packetBuffer, offset);
    CHECK(readThreadIdBytes == arm::pipe::ThreadIdSize);
    offset += uint8_t_size;

    uint32_t declarationSize = arm::pipe::ReadUint32(packetBuffer->GetReadableData(), offset);
    offset += uint32_t_size;
    for(uint32_t i = 0; i < declarationSize; ++i)
    {
        swTraceBufferMessages.push_back(arm::pipe::ReadSwTraceMessage(packetBuffer->GetReadableData(),
                                                                      offset,
                                                                      packetBuffer->GetSize()));
    }

    SendTimelinePacketToCommandHandler(packetBuffer->GetReadableData(), timelineDirectoryCaptureCommandHandler);

    for(uint32_t index = 0; index < declarationSize; ++index)
    {
        arm::pipe::SwTraceMessage& bufferMessage = swTraceBufferMessages[index];
        arm::pipe::SwTraceMessage& handlerMessage = timelineDirectoryCaptureCommandHandler.m_SwTraceMessages[index];

        CHECK(bufferMessage.m_Name == handlerMessage.m_Name);
        CHECK(bufferMessage.m_UiName == handlerMessage.m_UiName);
        CHECK(bufferMessage.m_Id == handlerMessage.m_Id);

        CHECK(bufferMessage.m_ArgTypes.size() == handlerMessage.m_ArgTypes.size());
        for(uint32_t i = 0; i < bufferMessage.m_ArgTypes.size(); ++i)
        {
            CHECK(bufferMessage.m_ArgTypes[i] == handlerMessage.m_ArgTypes[i]);
        }

        CHECK(bufferMessage.m_ArgNames.size() == handlerMessage.m_ArgNames.size());
        for(uint32_t i = 0; i < bufferMessage.m_ArgNames.size(); ++i)
        {
            CHECK(bufferMessage.m_ArgNames[i] == handlerMessage.m_ArgNames[i]);
        }
    }
}

TEST_CASE("TimelineCaptureTest")
{
    arm::pipe::BufferManager bufferManager(50);
    arm::pipe::TimelinePacketWriterFactory timelinePacketWriterFactory(bufferManager);

    std::unique_ptr<arm::pipe::ISendTimelinePacket> sendTimelinePacket =
        timelinePacketWriterFactory.GetSendTimelinePacket();

    arm::pipe::PacketVersionResolver packetVersionResolver;

    arm::pipe::TimelineDecoder timelineDecoder;

    arm::pipe::TimelineCaptureCommandHandler timelineCaptureCommandHandler(
        1, 1, packetVersionResolver.ResolvePacketVersion(1, 1).GetEncodedValue(), timelineDecoder,
        arm::pipe::ThreadIdSize);

    using Status = arm::pipe::ITimelineDecoder::TimelineStatus;
    CHECK(timelineDecoder.SetEntityCallback(PushEntity)             == Status::TimelineStatus_Success);
    CHECK(timelineDecoder.SetEventClassCallback(PushEventClass)     == Status::TimelineStatus_Success);
    CHECK(timelineDecoder.SetEventCallback(PushEvent)               == Status::TimelineStatus_Success);
    CHECK(timelineDecoder.SetLabelCallback(PushLabel)               == Status::TimelineStatus_Success);
    CHECK(timelineDecoder.SetRelationshipCallback(PushRelationship) == Status::TimelineStatus_Success);

    const uint64_t entityGuid = 111111u;
    const uint64_t eventClassGuid = 22222u;
    const uint64_t eventClassNameGuid = 22322u;
    const uint64_t timestamp = 33333u;
    const uint64_t eventGuid = 44444u;

    const int threadId = arm::pipe::GetCurrentThreadId();

    // need to do a bit of work here to extract the value from threadId
    unsigned char* uCharThreadId = new unsigned char[arm::pipe::ThreadIdSize]();;
    uint64_t uint64ThreadId;

    arm::pipe::WriteBytes(uCharThreadId, 0, &threadId, arm::pipe::ThreadIdSize);

    if (arm::pipe::ThreadIdSize == 4)
    {
        uint64ThreadId =  arm::pipe::ReadUint32(uCharThreadId, 0);
    }
    else if (arm::pipe::ThreadIdSize == 8)
    {
        uint64ThreadId =  arm::pipe::ReadUint64(uCharThreadId, 0);
    }
    delete[] uCharThreadId;

    const uint64_t labelGuid = 66666u;
    std::string labelName = "test_label";

    const uint64_t relationshipGuid = 77777u;
    const uint64_t headGuid = 888888u;
    const uint64_t tailGuid = 999999u;

    for (int i = 0; i < 10; ++i)
    {
        // Send entity
        sendTimelinePacket->SendTimelineEntityBinaryPacket(entityGuid);
        sendTimelinePacket->Commit();
        SendTimelinePacketToCommandHandler(bufferManager.GetReadableBuffer()->GetReadableData(),
                                           timelineCaptureCommandHandler);

        // Send event class
        sendTimelinePacket->SendTimelineEventClassBinaryPacket(eventClassGuid, eventClassNameGuid);
        sendTimelinePacket->Commit();
        SendTimelinePacketToCommandHandler(bufferManager.GetReadableBuffer()->GetReadableData(),
                                           timelineCaptureCommandHandler);

        // Send event
        sendTimelinePacket->SendTimelineEventBinaryPacket(timestamp, threadId, eventGuid);
        sendTimelinePacket->Commit();
        SendTimelinePacketToCommandHandler(bufferManager.GetReadableBuffer()->GetReadableData(),
                                           timelineCaptureCommandHandler);

        // Send label
        sendTimelinePacket->SendTimelineLabelBinaryPacket(labelGuid, labelName);
        sendTimelinePacket->Commit();
        SendTimelinePacketToCommandHandler(bufferManager.GetReadableBuffer()->GetReadableData(),
                                           timelineCaptureCommandHandler);

        // Send relationship
        arm::pipe::ProfilingRelationshipType relationshipType =
            arm::pipe::ProfilingRelationshipType::DataLink;
        sendTimelinePacket->SendTimelineRelationshipBinaryPacket(relationshipType,
                                                                 relationshipGuid,
                                                                 headGuid,
                                                                 tailGuid,
                                                                 0);
        sendTimelinePacket->Commit();
        SendTimelinePacketToCommandHandler(bufferManager.GetReadableBuffer()->GetReadableData(),
                                           timelineCaptureCommandHandler);
    }

    timelineDecoder.ApplyToModel([&](const arm::pipe::TimelineDecoder::Model& model){
        for (unsigned long i = 0; i < 10; ++i)
        {
            CHECK(model.m_Entities[i].m_Guid == entityGuid);

            CHECK(model.m_EventClasses[i].m_Guid == eventClassGuid);

            CHECK(model.m_Events[i].m_TimeStamp == timestamp);
            CHECK(model.m_Events[i].m_ThreadId == uint64ThreadId);
            CHECK(model.m_Events[i].m_Guid == eventGuid);

            CHECK(model.m_Labels[i].m_Guid == labelGuid);
            CHECK(model.m_Labels[i].m_Name == labelName);

            CHECK(model.m_Relationships[i].m_RelationshipType ==
                arm::pipe::ITimelineDecoder::RelationshipType::DataLink);
            CHECK(model.m_Relationships[i].m_Guid == relationshipGuid);
            CHECK(model.m_Relationships[i].m_HeadGuid == headGuid);
            CHECK(model.m_Relationships[i].m_TailGuid == tailGuid);
        }
    });
}

TEST_CASE("TimelineCaptureTestMultipleStringsInBuffer")
{
    arm::pipe::BufferManager               bufferManager(50);
    arm::pipe::TimelinePacketWriterFactory timelinePacketWriterFactory(bufferManager);

    std::unique_ptr<arm::pipe::ISendTimelinePacket> sendTimelinePacket =
                                                        timelinePacketWriterFactory.GetSendTimelinePacket();

    arm::pipe::PacketVersionResolver packetVersionResolver;

    arm::pipe::TimelineDecoder timelineDecoder;

    arm::pipe::TimelineCaptureCommandHandler timelineCaptureCommandHandler(
        1, 1, packetVersionResolver.ResolvePacketVersion(1, 1).GetEncodedValue(), timelineDecoder,
        arm::pipe::ThreadIdSize);

    using Status = arm::pipe::TimelineDecoder::TimelineStatus;
    CHECK(timelineDecoder.SetEntityCallback(PushEntity) == Status::TimelineStatus_Success);
    CHECK(timelineDecoder.SetEventClassCallback(PushEventClass) == Status::TimelineStatus_Success);
    CHECK(timelineDecoder.SetEventCallback(PushEvent) == Status::TimelineStatus_Success);
    CHECK(timelineDecoder.SetLabelCallback(PushLabel) == Status::TimelineStatus_Success);
    CHECK(timelineDecoder.SetRelationshipCallback(PushRelationship) == Status::TimelineStatus_Success);

    const uint64_t entityGuid         = 111111u;
    const uint64_t eventClassGuid     = 22222u;
    const uint64_t eventClassNameGuid = 22322u;
    const uint64_t timestamp          = 33333u;
    const uint64_t eventGuid          = 44444u;

    const int threadId = arm::pipe::GetCurrentThreadId();

    // need to do a bit of work here to extract the value from threadId
    unsigned char* uCharThreadId = new unsigned char[arm::pipe::ThreadIdSize]();
    uint64_t uint64ThreadId;

    arm::pipe::WriteBytes(uCharThreadId, 0, &threadId, arm::pipe::ThreadIdSize);

    if ( arm::pipe::ThreadIdSize == 4 )
    {
        uint64ThreadId = arm::pipe::ReadUint32(uCharThreadId, 0);
    }
    else if ( arm::pipe::ThreadIdSize == 8 )
    {
        uint64ThreadId = arm::pipe::ReadUint64(uCharThreadId, 0);
    }
    delete[] uCharThreadId;

    const uint64_t labelGuid  = 66666u;
    std::string    labelName  = "test_label";
    std::string    labelName2 = "test_label2";
    std::string    labelName3 = "test_label32";

    const uint64_t relationshipGuid = 77777u;
    const uint64_t headGuid         = 888888u;
    const uint64_t tailGuid         = 999999u;

    // Check with multiple messages in the same buffer
    for ( int i = 0; i < 9; ++i )
    {
        // Send entity
        sendTimelinePacket->SendTimelineEntityBinaryPacket(entityGuid);
        // Send event class
        sendTimelinePacket->SendTimelineEventClassBinaryPacket(eventClassGuid, eventClassNameGuid);
        // Send event
        sendTimelinePacket->SendTimelineEventBinaryPacket(timestamp, threadId, eventGuid);
        // Send label
        sendTimelinePacket->SendTimelineLabelBinaryPacket(labelGuid, labelName);
        sendTimelinePacket->SendTimelineLabelBinaryPacket(labelGuid, labelName2);
        sendTimelinePacket->SendTimelineLabelBinaryPacket(labelGuid, labelName3);
        // Send relationship
        arm::pipe::ProfilingRelationshipType relationshipType =
            arm::pipe::ProfilingRelationshipType::DataLink;
        sendTimelinePacket->SendTimelineRelationshipBinaryPacket(relationshipType,
                                                                 relationshipGuid,
                                                                 headGuid,
                                                                 tailGuid,
                                                                 0);
    }

    sendTimelinePacket->Commit();
    SendTimelinePacketToCommandHandler(bufferManager.GetReadableBuffer()->GetReadableData(),
                                       timelineCaptureCommandHandler);

    timelineDecoder.ApplyToModel([&](const arm::pipe::TimelineDecoder::Model& model){
        for ( unsigned long i = 0; i < 9; ++i )
        {
            CHECK(model.m_Entities[i].m_Guid == entityGuid);

            CHECK(model.m_EventClasses[i].m_Guid == eventClassGuid);

            CHECK(model.m_Labels[i].m_Guid == labelGuid);

            CHECK(model.m_Events[i].m_TimeStamp == timestamp);
            CHECK(model.m_Events[i].m_ThreadId == uint64ThreadId);
            CHECK(model.m_Events[i].m_Guid == eventGuid);

            CHECK(model.m_Relationships[i].m_RelationshipType ==
                arm::pipe::ITimelineDecoder::RelationshipType::DataLink);
            CHECK(model.m_Relationships[i].m_Guid == relationshipGuid);
            CHECK(model.m_Relationships[i].m_HeadGuid == headGuid);
            CHECK(model.m_Relationships[i].m_TailGuid == tailGuid);
        }
        for ( unsigned long i = 0; i < 9; i += 3 )
        {
            CHECK(model.m_Labels[i].m_Name == labelName);
            CHECK(model.m_Labels[i+1].m_Name == labelName2);
            CHECK(model.m_Labels[i+2].m_Name == labelName3);
        }
    });
}

}
