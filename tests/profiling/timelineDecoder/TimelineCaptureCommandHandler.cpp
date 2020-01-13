//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TimelineCaptureCommandHandler.hpp"

#include <iostream>
#include <string>

namespace armnn
{

namespace gatordmock
{

//Array of member functions, the array index matches the decl_id
const TimelineCaptureCommandHandler::ReadFunction TimelineCaptureCommandHandler::m_ReadFunctions[5]
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

    if (packet.GetLength() < 8)
    {
        return;
    }

    const unsigned char* data = reinterpret_cast<const unsigned char*>(packet.GetData());

    uint32_t declId = 0;

    declId = profiling::ReadUint32(data, offset);
    offset += uint32_t_size;

    (this->*m_ReadFunctions[declId])(data, offset);
}

void TimelineCaptureCommandHandler::ReadLabel(const unsigned char* data, uint32_t offset)
{
    Label label;
    label.m_Guid = profiling::ReadUint64(data, offset);
    offset += uint64_t_size;

    uint32_t nameLength = profiling::ReadUint32(data, offset);
    offset += uint32_t_size;

    label.m_Name = new char[nameLength];
    for (uint32_t i = 0; i< nameLength; ++i)
    {
        label.m_Name[i] = static_cast<char>(profiling::ReadUint8(data, offset + i));
    }

    CreateLabel(label, m_Model);

    if (!m_QuietOperation)
    {
        printLabels();
    }
}

void TimelineCaptureCommandHandler::ReadEntity(const unsigned char* data, uint32_t offset)
{
    Entity entity;
    entity.m_Guid = profiling::ReadUint64(data, offset);

    CreateEntity(entity, m_Model);

    if (!m_QuietOperation)
    {
        printEntities();
    }
}

void TimelineCaptureCommandHandler::ReadEventClass(const unsigned char* data, uint32_t offset)
{
    EventClass eventClass;
    eventClass.m_Guid = profiling::ReadUint64(data, offset);

    CreateEventClass(eventClass, m_Model);

    if (!m_QuietOperation)
    {
        printEventClasses();
    }
}

void TimelineCaptureCommandHandler::ReadRelationship(const unsigned char* data, uint32_t offset)
{
    Relationship relationship;
    relationship.m_RelationshipType = static_cast<RelationshipType>(profiling::ReadUint32(data, offset));
    offset += uint32_t_size;

    relationship.m_Guid = profiling::ReadUint64(data, offset);
    offset += uint64_t_size;

    relationship.m_HeadGuid  = profiling::ReadUint64(data, offset);
    offset += uint64_t_size;

    relationship.m_TailGuid = profiling::ReadUint64(data, offset);

    CreateRelationship(relationship, m_Model);

    if (!m_QuietOperation)
    {
        printRelationships();
    }
}



void TimelineCaptureCommandHandler::ReadEvent(const unsigned char* data, uint32_t offset)
{
    Event event;
    event.m_TimeStamp = profiling::ReadUint64(data, offset);
    offset += uint64_t_size;

    event.m_ThreadId = new uint8_t[threadId_size];
    profiling::ReadBytes(data, offset, threadId_size, event.m_ThreadId);
    offset += threadId_size;

    event.m_Guid = profiling::ReadUint64(data, offset);

    CreateEvent(event, m_Model);

    if (!m_QuietOperation)
    {
        printEvents();
    }
}

void TimelineCaptureCommandHandler::operator()(const profiling::Packet& packet)
{
    ParseData(packet);
}

void TimelineCaptureCommandHandler::printLabels()
{
    std::string header;

    header.append(profiling::CentreAlignFormatting("guid", 12));
    header.append(" | ");
    header.append(profiling::CentreAlignFormatting("value", 30));
    header.append("\n");

    std::cout << "\n" << "\n";
    std::cout << profiling::CentreAlignFormatting("LABELS", static_cast<int>(header.size()));
    std::cout << "\n";
    std::cout << std::string(header.size(), '=') << "\n";
    std::cout << header;

    for (uint32_t i = 0; i < m_Model->m_LabelCount; ++i)
    {
        std::string body;

        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model->m_Labels[i]->m_Guid), 12));
        body.append(" | ");
        body.append(profiling::CentreAlignFormatting(m_Model->m_Labels[i]->m_Name, 30));
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout<< body;
    }
}

void TimelineCaptureCommandHandler::printEntities()
{
    std::string header;
    header.append(profiling::CentreAlignFormatting("guid", 12));
    header.append("\n");

    std::cout << "\n" << "\n";
    std::cout << profiling::CentreAlignFormatting("ENTITIES", static_cast<int>(header.size()));
    std::cout << "\n";
    std::cout << std::string(header.size(), '=') << "\n";
    std::cout << header;

    for (uint32_t i = 0; i < m_Model->m_EntityCount; ++i)
    {
        std::string body;

        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model->m_Entities[i]->m_Guid), 12));
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout<< body;
    }
}

void TimelineCaptureCommandHandler::printEventClasses()
{
    std::string header;
    header.append(profiling::CentreAlignFormatting("guid", 12));
    header.append("\n");

    std::cout << "\n" << "\n";
    std::cout << profiling::CentreAlignFormatting("EVENT CLASSES", static_cast<int>(header.size()));
    std::cout << "\n";
    std::cout << std::string(header.size(), '=') << "\n";
    std::cout << header;

    for (uint32_t i = 0; i < m_Model->m_EventClassCount; ++i)
    {
        std::string body;

        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model->m_EventClasses[i]->m_Guid), 12));
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout<< body;
    }
}

void TimelineCaptureCommandHandler::printRelationships()
{
    std::string header;
    header.append(profiling::CentreAlignFormatting("relationshipType", 20));
    header.append(" | ");
    header.append(profiling::CentreAlignFormatting("relationshipGuid", 20));
    header.append(" | ");
    header.append(profiling::CentreAlignFormatting("headGuid", 12));
    header.append(" | ");
    header.append(profiling::CentreAlignFormatting("tailGuid", 12));
    header.append("\n");

    std::cout << "\n" << "\n";
    std::cout << profiling::CentreAlignFormatting("RELATIONSHIPS", static_cast<int>(header.size()));
    std::cout << "\n";
    std::cout << std::string(header.size(), '=') << "\n";
    std::cout << header;

    for (uint32_t i = 0; i < m_Model->m_RelationshipCount; ++i)
    {
        std::string body;

        body.append(
                profiling::CentreAlignFormatting(std::to_string(m_Model->m_Relationships[i]->m_RelationshipType), 20));
        body.append(" | ");
        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model->m_Relationships[i]->m_Guid), 20));
        body.append(" | ");
        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model->m_Relationships[i]->m_HeadGuid), 12));
        body.append(" | ");
        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model->m_Relationships[i]->m_TailGuid), 12));
        body.append(" | ");
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout<< body;
    }
}

void TimelineCaptureCommandHandler::printEvents()
{
    std::string header;

    header.append(profiling::CentreAlignFormatting("timestamp", 12));
    header.append(" | ");
    header.append(profiling::CentreAlignFormatting("threadId", 12));
    header.append(" | ");
    header.append(profiling::CentreAlignFormatting("eventGuid", 12));
    header.append("\n");

    std::cout << "\n" << "\n";
    std::cout << profiling::CentreAlignFormatting("EVENTS", static_cast<int>(header.size()));
    std::cout << "\n";
    std::cout << std::string(header.size(), '=') << "\n";
    std::cout << header;

    for (uint32_t i = 0; i < m_Model->m_EventCount; ++i)
    {
        std::string body;

        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model->m_Events[i]->m_TimeStamp), 12));
        body.append(" | ");

        std::string threadId;
        for(uint32_t j =0; j< threadId_size; j++)
        {
            threadId += static_cast<char>(m_Model->m_Events[i]->m_ThreadId[j]);
        }
        body.append(profiling::CentreAlignFormatting(threadId, 12));
        body.append(" | ");
        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model->m_Events[i]->m_Guid), 12));
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout<< body;
    }
}

} //namespace gatordmock

} //namespace armnn
