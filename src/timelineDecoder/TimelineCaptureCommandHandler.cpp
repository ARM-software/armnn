//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TimelineCaptureCommandHandler.hpp"

#include <string>

namespace armnn
{

namespace timelinedecoder
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

void TimelineCaptureCommandHandler::ParseData(const armnn::profiling::Packet& packet)
{
    uint32_t offset = 0;
    m_PacketLength = packet.GetLength();

    if ( m_PacketLength < 8 )
    {
        return;
    }

    const unsigned char* data = reinterpret_cast<const unsigned char*>(packet.GetData());

    uint32_t declId = 0;

    while ( offset < m_PacketLength )
    {
        declId = profiling::ReadUint32(data, offset);
        offset += uint32_t_size;

        (this->*m_ReadFunctions[declId])(data, offset);
    }
}

void TimelineCaptureCommandHandler::ReadLabel(const unsigned char* data, uint32_t& offset)
{
    ITimelineDecoder::Label label;
    label.m_Guid = profiling::ReadUint64(data, offset);
    offset += uint64_t_size;

    uint32_t nameLength = profiling::ReadUint32(data, offset);
    offset += uint32_t_size;

    uint32_t i = 0;
    // nameLength - 1 to account for null operator \0
    for ( i = 0; i < nameLength - 1; ++i )
    {
        label.m_Name += static_cast<char>(profiling::ReadUint8(data, offset + i));
    }
    // Shift offset past nameLength
    uint32_t uint32WordAmount = (nameLength / uint32_t_size) + (nameLength % uint32_t_size != 0 ? 1 : 0);
    offset += uint32WordAmount * uint32_t_size;

    m_TimelineDecoder.CreateLabel(label);
}

void TimelineCaptureCommandHandler::ReadEntity(const unsigned char* data, uint32_t& offset)
{
    ITimelineDecoder::Entity entity;
    entity.m_Guid = profiling::ReadUint64(data, offset);
    offset += uint64_t_size;
    m_TimelineDecoder.CreateEntity(entity);
}

void TimelineCaptureCommandHandler::ReadEventClass(const unsigned char* data, uint32_t& offset)
{
    ITimelineDecoder::EventClass eventClass;
    eventClass.m_Guid = profiling::ReadUint64(data, offset);
    offset += uint64_t_size;
    m_TimelineDecoder.CreateEventClass(eventClass);
}

void TimelineCaptureCommandHandler::ReadRelationship(const unsigned char* data, uint32_t& offset)
{
    ITimelineDecoder::Relationship relationship;
    relationship.m_RelationshipType =
        static_cast<ITimelineDecoder::RelationshipType>(profiling::ReadUint32(data, offset));
    offset += uint32_t_size;

    relationship.m_Guid = profiling::ReadUint64(data, offset);
    offset += uint64_t_size;

    relationship.m_HeadGuid = profiling::ReadUint64(data, offset);
    offset += uint64_t_size;

    relationship.m_TailGuid = profiling::ReadUint64(data, offset);
    offset += uint64_t_size;
    m_TimelineDecoder.CreateRelationship(relationship);
}

void TimelineCaptureCommandHandler::ReadEvent(const unsigned char* data, uint32_t& offset)
{
    ITimelineDecoder::Event event;
    event.m_TimeStamp = profiling::ReadUint64(data, offset);
    offset += uint64_t_size;

    if ( m_ThreadIdSize == 4 )
    {
        event.m_ThreadId = profiling::ReadUint32(data, offset);
    }
    else if ( m_ThreadIdSize == 8 )
    {
        event.m_ThreadId = profiling::ReadUint64(data, offset);
    }

    offset += m_ThreadIdSize;

    event.m_Guid = profiling::ReadUint64(data, offset);
    offset += uint64_t_size;

    m_TimelineDecoder.CreateEvent(event);
}

void TimelineCaptureCommandHandler::operator()(const profiling::Packet& packet)
{
    ParseData(packet);
}

} //namespace gatordmock

} //namespace armnn
