//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <common/include/CommonProfilingUtils.hpp>
#include <common/include/Logging.hpp>
#include <server/include/timelineDecoder/TimelineCaptureCommandHandler.hpp>

#include <string>

namespace arm
{

namespace pipe
{

//Array of member functions, the array index matches the decl_id
const TimelineCaptureCommandHandler::ReadFunction TimelineCaptureCommandHandler::m_ReadFunctions[]
{
    &TimelineCaptureCommandHandler::ReadLabel,              // Label decl_id = 0
    &TimelineCaptureCommandHandler::ReadEntity,             // Entity decl_id = 1
    &TimelineCaptureCommandHandler::ReadEventClass,         // EventClass decl_id = 2
    &TimelineCaptureCommandHandler::ReadRelationship,       // Relationship decl_id = 3
    &TimelineCaptureCommandHandler::ReadEvent               // Event decl_id = 4
};

void TimelineCaptureCommandHandler::SetThreadIdSize(uint32_t size)
{
    m_ThreadIdSize = size;
}

void TimelineCaptureCommandHandler::operator()(const arm::pipe::Packet& packet)
{
    ParseData(packet);
}

void TimelineCaptureCommandHandler::ParseData(const arm::pipe::Packet& packet)
{
    uint32_t offset = 0;
    m_PacketLength = packet.GetLength();

    // We are expecting TimelineDirectoryCaptureCommandHandler to set the thread id size
    // if it not set in the constructor
    if (m_ThreadIdSize == 0)
    {
        ARM_PIPE_LOG(error) << "TimelineCaptureCommandHandler: m_ThreadIdSize has not been set";
        return;
    }

    if (packet.GetLength() < 8)
    {
        return;
    }

    const unsigned char* data = reinterpret_cast<const unsigned char*>(packet.GetData());

    uint32_t declId = 0;

    while ( offset < m_PacketLength )
    {
        declId = arm::pipe::ReadUint32(data, offset);
        offset += uint32_t_size;

        ITimelineDecoder::TimelineStatus status = (this->*m_ReadFunctions[declId])(data, offset);
        if (status == ITimelineDecoder::TimelineStatus::TimelineStatus_Fail)
        {
            ARM_PIPE_LOG(error) << "Decode of timeline message type [" << declId <<
                                "] at offset [" << offset << "] failed";
            break;
        }
    }
}

ITimelineDecoder::TimelineStatus TimelineCaptureCommandHandler::ReadLabel(const unsigned char* data, uint32_t& offset)
{
    ITimelineDecoder::Label label;
    label.m_Guid = arm::pipe::ReadUint64(data, offset);
    offset += uint64_t_size;

    uint32_t nameLength = arm::pipe::ReadUint32(data, offset);
    offset += uint32_t_size;

    uint32_t i = 0;
    // nameLength - 1 to account for null operator \0
    for ( i = 0; i < nameLength - 1; ++i )
    {
        label.m_Name += static_cast<char>(arm::pipe::ReadUint8(data, offset + i));
    }
    // Shift offset past nameLength
    uint32_t uint32WordAmount = (nameLength / uint32_t_size) + (nameLength % uint32_t_size != 0 ? 1 : 0);
    offset += uint32WordAmount * uint32_t_size;

    return m_TimelineDecoder.CreateLabel(label);
}

ITimelineDecoder::TimelineStatus TimelineCaptureCommandHandler::ReadEntity(
        const unsigned char* data, uint32_t& offset)
{
    ITimelineDecoder::Entity entity;
    entity.m_Guid = arm::pipe::ReadUint64(data, offset);
    offset += uint64_t_size;
    return m_TimelineDecoder.CreateEntity(entity);
}

ITimelineDecoder::TimelineStatus TimelineCaptureCommandHandler::ReadEventClass(
    const unsigned char* data, uint32_t& offset)
{
    ITimelineDecoder::EventClass eventClass;
    eventClass.m_Guid = arm::pipe::ReadUint64(data, offset);
    offset += uint64_t_size;
    eventClass.m_NameGuid = arm::pipe::ReadUint64(data, offset);
    offset += uint64_t_size;
    return m_TimelineDecoder.CreateEventClass(eventClass);
}

ITimelineDecoder::TimelineStatus TimelineCaptureCommandHandler::ReadRelationship(
    const unsigned char* data, uint32_t& offset)
{
    ITimelineDecoder::Relationship relationship;
    relationship.m_RelationshipType =
        static_cast<ITimelineDecoder::RelationshipType>(arm::pipe::ReadUint32(data, offset));
    offset += uint32_t_size;

    relationship.m_Guid = arm::pipe::ReadUint64(data, offset);
    offset += uint64_t_size;

    relationship.m_HeadGuid = arm::pipe::ReadUint64(data, offset);
    offset += uint64_t_size;

    relationship.m_TailGuid = arm::pipe::ReadUint64(data, offset);
    offset += uint64_t_size;

    relationship.m_AttributeGuid = arm::pipe::ReadUint64(data, offset);
    offset += uint64_t_size;

    return m_TimelineDecoder.CreateRelationship(relationship);
}

ITimelineDecoder::TimelineStatus TimelineCaptureCommandHandler::ReadEvent(
    const unsigned char* data, uint32_t& offset)
{
    ITimelineDecoder::Event event;
    event.m_TimeStamp = arm::pipe::ReadUint64(data, offset);
    offset += uint64_t_size;

    if ( m_ThreadIdSize == 4 )
    {
        event.m_ThreadId = arm::pipe::ReadUint32(data, offset);
    }
    else if ( m_ThreadIdSize == 8 )
    {
        event.m_ThreadId = arm::pipe::ReadUint64(data, offset);
    }

    offset += m_ThreadIdSize;

    event.m_Guid = arm::pipe::ReadUint64(data, offset);
    offset += uint64_t_size;

    return m_TimelineDecoder.CreateEvent(event);
}

} //namespace pipe

} //namespace arm
