//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <TimelineCaptureCommandHandler.hpp>
#include <TimelineDirectoryCaptureCommandHandler.hpp>
#include <TimelineDecoder.hpp>

#include <CommandHandlerFunctor.hpp>
#include <ProfilingService.hpp>
#include <PacketBuffer.hpp>
#include <TimelinePacketWriterFactory.hpp>

#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

BOOST_AUTO_TEST_SUITE(TimelineDecoderTests)

using namespace armnn;
using namespace timelinedecoder;

void SendTimelinePacketToCommandHandler(const unsigned char* packetBuffer,
                                        profiling::CommandHandlerFunctor &CommandHandler)
{
    uint32_t uint32_t_size = sizeof(uint32_t);
    unsigned int offset = 0;

    uint32_t header[2];
    header[0] = profiling::ReadUint32(packetBuffer, offset);
    offset += uint32_t_size;
    header[1] = profiling::ReadUint32(packetBuffer, offset);
    offset += uint32_t_size;
    uint32_t PacketDataLength  = header[1] & 0x00FFFFFF;

    auto uniquePacketData = std::make_unique<unsigned char[]>(PacketDataLength);
    std::memcpy(uniquePacketData.get(), packetBuffer + offset, PacketDataLength);

    armnn::profiling::Packet packet(header[0], PacketDataLength, uniquePacketData);

    BOOST_CHECK(std::memcmp(packetBuffer + offset, packet.GetData(), packet.GetLength()) == 0);

    CommandHandler(packet);
}

void PushEntity(TimelineDecoder::Model& model, const ITimelineDecoder::Entity entity)
{
    model.m_Entities.emplace_back(entity);
}

void PushEventClass(TimelineDecoder::Model& model, const ITimelineDecoder::EventClass eventClass)
{
    model.m_EventClasses.emplace_back(eventClass);
}

void PushEvent(TimelineDecoder::Model& model, const ITimelineDecoder::Event event)
{
    model.m_Events.emplace_back(event);
}

void PushLabel(TimelineDecoder::Model& model, const ITimelineDecoder::Label label)
{
    model.m_Labels.emplace_back(label);
}

void PushRelationship(TimelineDecoder::Model& model, const ITimelineDecoder::Relationship relationship)
{
    model.m_Relationships.emplace_back(relationship);
}

BOOST_AUTO_TEST_CASE(TimelineDirectoryTest)
{
    uint32_t uint8_t_size  = sizeof(uint8_t);
    uint32_t uint32_t_size = sizeof(uint32_t);
    uint32_t uint64_t_size = sizeof(uint64_t);

    profiling::BufferManager bufferManager(5);
    profiling::TimelinePacketWriterFactory timelinePacketWriterFactory(bufferManager);

    std::unique_ptr<profiling::ISendTimelinePacket> sendTimelinePacket =
            timelinePacketWriterFactory.GetSendTimelinePacket();

    profiling::PacketVersionResolver packetVersionResolver;

    TimelineDecoder timelineDecoder;
    TimelineCaptureCommandHandler timelineCaptureCommandHandler(
            1, 1, packetVersionResolver.ResolvePacketVersion(1, 1).GetEncodedValue(), timelineDecoder);

    TimelineDirectoryCaptureCommandHandler timelineDirectoryCaptureCommandHandler(
            1, 0, packetVersionResolver.ResolvePacketVersion(1, 0).GetEncodedValue(),
            timelineCaptureCommandHandler, true);

    sendTimelinePacket->SendTimelineMessageDirectoryPackage();
    sendTimelinePacket->Commit();

    std::vector<profiling::SwTraceMessage> swTraceBufferMessages;

    unsigned int offset = uint32_t_size * 2;

    std::unique_ptr<profiling::IPacketBuffer> packetBuffer = bufferManager.GetReadableBuffer();

    uint8_t readStreamVersion = ReadUint8(packetBuffer, offset);
    BOOST_CHECK(readStreamVersion == 4);
    offset += uint8_t_size;
    uint8_t readPointerBytes = ReadUint8(packetBuffer, offset);
    BOOST_CHECK(readPointerBytes == uint64_t_size);
    offset += uint8_t_size;
    uint8_t readThreadIdBytes = ReadUint8(packetBuffer, offset);
    BOOST_CHECK(readThreadIdBytes == armnn::profiling::ThreadIdSize);
    offset += uint8_t_size;

    uint32_t declarationSize = profiling::ReadUint32(packetBuffer, offset);
    offset += uint32_t_size;
    for(uint32_t i = 0; i < declarationSize; ++i)
    {
        swTraceBufferMessages.push_back(profiling::ReadSwTraceMessage(packetBuffer->GetReadableData(), offset));
    }

    SendTimelinePacketToCommandHandler(packetBuffer->GetReadableData(), timelineDirectoryCaptureCommandHandler);

    for(uint32_t index = 0; index < declarationSize; ++index)
    {
        profiling::SwTraceMessage& bufferMessage = swTraceBufferMessages[index];
        profiling::SwTraceMessage& handlerMessage = timelineDirectoryCaptureCommandHandler.m_SwTraceMessages[index];

        BOOST_CHECK(bufferMessage.m_Name == handlerMessage.m_Name);
        BOOST_CHECK(bufferMessage.m_UiName == handlerMessage.m_UiName);
        BOOST_CHECK(bufferMessage.m_Id == handlerMessage.m_Id);

        BOOST_CHECK(bufferMessage.m_ArgTypes.size() == handlerMessage.m_ArgTypes.size());
        for(uint32_t i = 0; i < bufferMessage.m_ArgTypes.size(); ++i)
        {
            BOOST_CHECK(bufferMessage.m_ArgTypes[i] == handlerMessage.m_ArgTypes[i]);
        }

        BOOST_CHECK(bufferMessage.m_ArgNames.size() == handlerMessage.m_ArgNames.size());
        for(uint32_t i = 0; i < bufferMessage.m_ArgNames.size(); ++i)
        {
            BOOST_CHECK(bufferMessage.m_ArgNames[i] == handlerMessage.m_ArgNames[i]);
        }
    }
}

BOOST_AUTO_TEST_CASE(TimelineCaptureTest)
{
    profiling::BufferManager bufferManager(50);
    profiling::TimelinePacketWriterFactory timelinePacketWriterFactory(bufferManager);

    std::unique_ptr<profiling::ISendTimelinePacket> sendTimelinePacket =
        timelinePacketWriterFactory.GetSendTimelinePacket();

    profiling::PacketVersionResolver packetVersionResolver;

    TimelineDecoder timelineDecoder;
    const TimelineDecoder::Model& model = timelineDecoder.GetModel();


    TimelineCaptureCommandHandler timelineCaptureCommandHandler(
        1, 1, packetVersionResolver.ResolvePacketVersion(1, 1).GetEncodedValue(), timelineDecoder,
        armnn::profiling::ThreadIdSize);

    using Status = ITimelineDecoder::TimelineStatus;
    BOOST_CHECK(timelineDecoder.SetEntityCallback(PushEntity)             == Status::TimelineStatus_Success);
    BOOST_CHECK(timelineDecoder.SetEventClassCallback(PushEventClass)     == Status::TimelineStatus_Success);
    BOOST_CHECK(timelineDecoder.SetEventCallback(PushEvent)               == Status::TimelineStatus_Success);
    BOOST_CHECK(timelineDecoder.SetLabelCallback(PushLabel)               == Status::TimelineStatus_Success);
    BOOST_CHECK(timelineDecoder.SetRelationshipCallback(PushRelationship) == Status::TimelineStatus_Success);

    const uint64_t entityGuid = 111111u;
    const uint64_t eventClassGuid = 22222u;
    const uint64_t eventClassNameGuid = 22322u;
    const uint64_t timestamp = 33333u;
    const uint64_t eventGuid = 44444u;

    const std::thread::id threadId = std::this_thread::get_id();

    // need to do a bit of work here to extract the value from threadId
    unsigned char* uCharThreadId = new unsigned char[armnn::profiling::ThreadIdSize]();;
    uint64_t uint64ThreadId;

    profiling::WriteBytes(uCharThreadId, 0, &threadId, armnn::profiling::ThreadIdSize);

    if (armnn::profiling::ThreadIdSize == 4)
    {
        uint64ThreadId =  profiling::ReadUint32(uCharThreadId, 0);
    }
    else if (armnn::profiling::ThreadIdSize == 8)
    {
        uint64ThreadId =  profiling::ReadUint64(uCharThreadId, 0);
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
        profiling::ProfilingRelationshipType relationshipType = profiling::ProfilingRelationshipType::DataLink;
        sendTimelinePacket->SendTimelineRelationshipBinaryPacket(relationshipType,
                                                                 relationshipGuid,
                                                                 headGuid,
                                                                 tailGuid,
                                                                 0);
        sendTimelinePacket->Commit();
        SendTimelinePacketToCommandHandler(bufferManager.GetReadableBuffer()->GetReadableData(),
                                           timelineCaptureCommandHandler);
    }

    for (unsigned long i = 0; i < 10; ++i)
    {
        BOOST_CHECK(model.m_Entities[i].m_Guid == entityGuid);

        BOOST_CHECK(model.m_EventClasses[i].m_Guid == eventClassGuid);

        BOOST_CHECK(model.m_Events[i].m_TimeStamp == timestamp);
        BOOST_CHECK(model.m_Events[i].m_ThreadId == uint64ThreadId);
        BOOST_CHECK(model.m_Events[i].m_Guid == eventGuid);

        BOOST_CHECK(model.m_Labels[i].m_Guid == labelGuid);
        BOOST_CHECK(model.m_Labels[i].m_Name == labelName);

        BOOST_CHECK(model.m_Relationships[i].m_RelationshipType == ITimelineDecoder::RelationshipType::DataLink);
        BOOST_CHECK(model.m_Relationships[i].m_Guid == relationshipGuid);
        BOOST_CHECK(model.m_Relationships[i].m_HeadGuid == headGuid);
        BOOST_CHECK(model.m_Relationships[i].m_TailGuid == tailGuid);
    }
}

BOOST_AUTO_TEST_CASE(TimelineCaptureTestMultipleStringsInBuffer)
{
    profiling::BufferManager               bufferManager(50);
    profiling::TimelinePacketWriterFactory timelinePacketWriterFactory(bufferManager);

    std::unique_ptr<profiling::ISendTimelinePacket> sendTimelinePacket =
                                                        timelinePacketWriterFactory.GetSendTimelinePacket();

    profiling::PacketVersionResolver packetVersionResolver;

    TimelineDecoder timelineDecoder;
    const TimelineDecoder::Model& model = timelineDecoder.GetModel();

    TimelineCaptureCommandHandler timelineCaptureCommandHandler(
        1, 1, packetVersionResolver.ResolvePacketVersion(1, 1).GetEncodedValue(), timelineDecoder,
        armnn::profiling::ThreadIdSize);

    using Status = ITimelineDecoder::TimelineStatus;
    BOOST_CHECK(timelineDecoder.SetEntityCallback(PushEntity) == Status::TimelineStatus_Success);
    BOOST_CHECK(timelineDecoder.SetEventClassCallback(PushEventClass) == Status::TimelineStatus_Success);
    BOOST_CHECK(timelineDecoder.SetEventCallback(PushEvent) == Status::TimelineStatus_Success);
    BOOST_CHECK(timelineDecoder.SetLabelCallback(PushLabel) == Status::TimelineStatus_Success);
    BOOST_CHECK(timelineDecoder.SetRelationshipCallback(PushRelationship) == Status::TimelineStatus_Success);

    const uint64_t entityGuid         = 111111u;
    const uint64_t eventClassGuid     = 22222u;
    const uint64_t eventClassNameGuid = 22322u;
    const uint64_t timestamp          = 33333u;
    const uint64_t eventGuid          = 44444u;

    const std::thread::id threadId = std::this_thread::get_id();

    // need to do a bit of work here to extract the value from threadId
    unsigned char* uCharThreadId = new unsigned char[armnn::profiling::ThreadIdSize]();
    uint64_t uint64ThreadId;

    profiling::WriteBytes(uCharThreadId, 0, &threadId, armnn::profiling::ThreadIdSize);

    if ( armnn::profiling::ThreadIdSize == 4 )
    {
        uint64ThreadId = profiling::ReadUint32(uCharThreadId, 0);
    } 
    else if ( armnn::profiling::ThreadIdSize == 8 )
    {
        uint64ThreadId = profiling::ReadUint64(uCharThreadId, 0);
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
        profiling::ProfilingRelationshipType relationshipType = profiling::ProfilingRelationshipType::DataLink;
        sendTimelinePacket->SendTimelineRelationshipBinaryPacket(relationshipType,
                                                                 relationshipGuid,
                                                                 headGuid,
                                                                 tailGuid,
                                                                 0);
    }

    sendTimelinePacket->Commit();
    SendTimelinePacketToCommandHandler(bufferManager.GetReadableBuffer()->GetReadableData(),
                                       timelineCaptureCommandHandler);

    for ( unsigned long i = 0; i < 9; ++i )
    {
        BOOST_CHECK(model.m_Entities[i].m_Guid == entityGuid);

        BOOST_CHECK(model.m_EventClasses[i].m_Guid == eventClassGuid);

        BOOST_CHECK(model.m_Labels[i].m_Guid == labelGuid);

        BOOST_CHECK(model.m_Events[i].m_TimeStamp == timestamp);
        BOOST_CHECK(model.m_Events[i].m_ThreadId == uint64ThreadId);
        BOOST_CHECK(model.m_Events[i].m_Guid == eventGuid);

        BOOST_CHECK(model.m_Relationships[i].m_RelationshipType == ITimelineDecoder::RelationshipType::DataLink);
        BOOST_CHECK(model.m_Relationships[i].m_Guid == relationshipGuid);
        BOOST_CHECK(model.m_Relationships[i].m_HeadGuid == headGuid);
        BOOST_CHECK(model.m_Relationships[i].m_TailGuid == tailGuid);
    }
    for ( unsigned long i = 0; i < 9; i += 3 )
    {
        BOOST_CHECK(model.m_Labels[i].m_Name == labelName);
        BOOST_CHECK(model.m_Labels[i+1].m_Name == labelName2);
        BOOST_CHECK(model.m_Labels[i+2].m_Name == labelName3);
    }
}

BOOST_AUTO_TEST_SUITE_END()
