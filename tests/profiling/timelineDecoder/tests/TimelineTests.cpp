//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../TimelineCaptureCommandHandler.hpp"
#include "../TimelineDirectoryCaptureCommandHandler.hpp"
#include "../ITimelineDecoder.h"
#include "../TimelineModel.h"
#include "TimelineTestFunctions.hpp"

#include <CommandHandlerFunctor.hpp>
#include <ProfilingService.hpp>
#include <PacketBuffer.hpp>
#include <TimelinePacketWriterFactory.hpp>

#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

BOOST_AUTO_TEST_SUITE(TimelineDecoderTests)

using namespace armnn;

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

BOOST_AUTO_TEST_CASE(TimelineDirectoryTest)
{
    uint32_t uint8_t_size  = sizeof(uint8_t);
    uint32_t uint32_t_size = sizeof(uint32_t);
    uint32_t uint64_t_size = sizeof(uint64_t);
    uint32_t threadId_size = sizeof(std::thread::id);

    profiling::BufferManager bufferManager(5);
    profiling::TimelinePacketWriterFactory timelinePacketWriterFactory(bufferManager);

    std::unique_ptr<profiling::ISendTimelinePacket> sendTimelinePacket =
            timelinePacketWriterFactory.GetSendTimelinePacket();

    profiling::PacketVersionResolver packetVersionResolver;

    gatordmock::TimelineDirectoryCaptureCommandHandler timelineDirectoryCaptureCommandHandler(
            1, 0, packetVersionResolver.ResolvePacketVersion(1, 0).GetEncodedValue(), true);

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
    BOOST_CHECK(readThreadIdBytes == threadId_size);
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
    uint32_t threadId_size = sizeof(std::thread::id);

    profiling::BufferManager bufferManager(50);
    profiling::TimelinePacketWriterFactory timelinePacketWriterFactory(bufferManager);

    std::unique_ptr<profiling::ISendTimelinePacket> sendTimelinePacket =
        timelinePacketWriterFactory.GetSendTimelinePacket();

    profiling::PacketVersionResolver packetVersionResolver;

    Model* modelPtr;
    CreateModel(&modelPtr);

    gatordmock::TimelineCaptureCommandHandler timelineCaptureCommandHandler(
        1, 1, packetVersionResolver.ResolvePacketVersion(1, 1).GetEncodedValue(), modelPtr, true);

    BOOST_CHECK(SetEntityCallback(PushEntity, modelPtr)             == ErrorCode_Success);
    BOOST_CHECK(SetEventClassCallback(PushEventClass, modelPtr)     == ErrorCode_Success);
    BOOST_CHECK(SetEventCallback(PushEvent, modelPtr)               == ErrorCode_Success);
    BOOST_CHECK(SetLabelCallback(PushLabel, modelPtr)               == ErrorCode_Success);
    BOOST_CHECK(SetRelationshipCallback(PushRelationship, modelPtr) == ErrorCode_Success);

    const uint64_t entityGuid = 22222u;

    const uint64_t eventClassGuid = 33333u;

    const uint64_t timestamp = 111111u;
    const uint64_t eventGuid = 55555u;

    const std::thread::id threadId = std::this_thread::get_id();;

    const uint64_t labelGuid = 11111u;
    std::string labelName = "test_label";

    const uint64_t relationshipGuid = 44444u;
    const uint64_t headGuid = 111111u;
    const uint64_t tailGuid = 222222u;

    for (int i = 0; i < 10; ++i)
    {
        // Send entity
        sendTimelinePacket->SendTimelineEntityBinaryPacket(entityGuid);
        sendTimelinePacket->Commit();
        SendTimelinePacketToCommandHandler(bufferManager.GetReadableBuffer()->GetReadableData(),
                                           timelineCaptureCommandHandler);

        // Send event class
        sendTimelinePacket->SendTimelineEventClassBinaryPacket(eventClassGuid);
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
                                                                 tailGuid);
        sendTimelinePacket->Commit();
        SendTimelinePacketToCommandHandler(bufferManager.GetReadableBuffer()->GetReadableData(),
                                           timelineCaptureCommandHandler);
    }

    for (int i = 0; i < 10; ++i)
    {
        BOOST_CHECK(modelPtr->m_Entities[i]->m_Guid == entityGuid);

        BOOST_CHECK(modelPtr->m_EventClasses[i]->m_Guid == eventClassGuid);

        BOOST_CHECK(modelPtr->m_Events[i]->m_TimeStamp == timestamp);

        std::vector<uint8_t> readThreadId(threadId_size, 0);
        profiling::ReadBytes(modelPtr->m_Events[i]->m_ThreadId, 0, threadId_size, readThreadId.data());
        BOOST_CHECK(readThreadId == threadId);

        BOOST_CHECK(modelPtr->m_Events[i]->m_Guid == eventGuid);

        BOOST_CHECK(modelPtr->m_Labels[i]->m_Guid == labelGuid);
        BOOST_CHECK(std::string(modelPtr->m_Labels[i]->m_Name) == labelName);

        BOOST_CHECK(modelPtr->m_Relationships[i]->m_RelationshipType == RelationshipType::DataLink);
        BOOST_CHECK(modelPtr->m_Relationships[i]->m_Guid == relationshipGuid);
        BOOST_CHECK(modelPtr->m_Relationships[i]->m_HeadGuid == headGuid);
        BOOST_CHECK(modelPtr->m_Relationships[i]->m_TailGuid == tailGuid);
    }

    DestroyModel(&modelPtr);
}

BOOST_AUTO_TEST_SUITE_END()
