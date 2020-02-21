//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TimelineDecoder.hpp"

#include <ProfilingUtils.hpp>

#include <iostream>
namespace armnn
{
TimelineDecoder::ErrorCode TimelineDecoder::CreateEntity(const Entity &entity)
{
    if (m_OnNewEntityCallback == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    m_OnNewEntityCallback(m_Model, entity);

    return ErrorCode::ErrorCode_Success;
}

TimelineDecoder::ErrorCode TimelineDecoder::CreateEventClass(const EventClass &eventClass)
{
    if (m_OnNewEventClassCallback == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    m_OnNewEventClassCallback(m_Model, eventClass);

    return ErrorCode::ErrorCode_Success;
}

TimelineDecoder::ErrorCode TimelineDecoder::CreateEvent(const Event &event)
{
    if (m_OnNewEventCallback == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    m_OnNewEventCallback(m_Model, event);

    return ErrorCode::ErrorCode_Success;
}

TimelineDecoder::ErrorCode TimelineDecoder::CreateLabel(const Label &label)
{
    if (m_OnNewLabelCallback == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    m_OnNewLabelCallback(m_Model, label);

    return ErrorCode::ErrorCode_Success;
}

TimelineDecoder::ErrorCode TimelineDecoder::CreateRelationship(const Relationship &relationship)
{
    if (m_OnNewRelationshipCallback == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    m_OnNewRelationshipCallback(m_Model, relationship);
    return ErrorCode::ErrorCode_Success;
}

const TimelineDecoder::Model &TimelineDecoder::GetModel()
{
    return m_Model;
}

TimelineDecoder::ErrorCode TimelineDecoder::SetEntityCallback(OnNewEntityCallback cb)
{
    if (cb == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    m_OnNewEntityCallback = cb;
    return ErrorCode::ErrorCode_Success;
}

TimelineDecoder::ErrorCode TimelineDecoder::SetEventClassCallback(OnNewEventClassCallback cb)
{
    if (cb == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    m_OnNewEventClassCallback = cb;
    return ErrorCode::ErrorCode_Success;
}

TimelineDecoder::ErrorCode TimelineDecoder::SetEventCallback(OnNewEventCallback cb)
{
    if (cb == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    m_OnNewEventCallback = cb;
    return ErrorCode::ErrorCode_Success;
}

TimelineDecoder::ErrorCode TimelineDecoder::SetLabelCallback(OnNewLabelCallback cb)
{
    if (cb == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    m_OnNewLabelCallback = cb;
    return ErrorCode::ErrorCode_Success;
}

TimelineDecoder::ErrorCode TimelineDecoder::SetRelationshipCallback(OnNewRelationshipCallback cb)
{
    if (cb == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    m_OnNewRelationshipCallback = cb;
    return ErrorCode::ErrorCode_Success;
}

void TimelineDecoder::print()
{
    printLabels();
    printEntities();
    printEventClasses();
    printEvents();
    printRelationships();
}

void TimelineDecoder::printLabels()
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

    for (uint32_t i = 0; i < m_Model.m_Labels.size(); ++i)
    {
        std::string body;

        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model.m_Labels[i].m_Guid), 12));
        body.append(" | ");
        body.append(profiling::CentreAlignFormatting(m_Model.m_Labels[i].m_Name, 30));
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout << body;
    }
}

void TimelineDecoder::printEntities()
{
    std::string header;
    header.append(profiling::CentreAlignFormatting("guid", 12));
    header.append("\n");

    std::cout << "\n" << "\n";
    std::cout << profiling::CentreAlignFormatting("ENTITIES", static_cast<int>(header.size()));
    std::cout << "\n";
    std::cout << std::string(header.size(), '=') << "\n";
    std::cout << header;

    for (uint32_t i = 0; i < m_Model.m_Entities.size(); ++i)
    {
        std::string body;

        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model.m_Entities[i].m_Guid), 12));
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout << body;
    }
}

void TimelineDecoder::printEventClasses()
{
    std::string header;
    header.append(profiling::CentreAlignFormatting("guid", 12));
    header.append("\n");

    std::cout << "\n" << "\n";
    std::cout << profiling::CentreAlignFormatting("EVENT CLASSES", static_cast<int>(header.size()));
    std::cout << "\n";
    std::cout << std::string(header.size(), '=') << "\n";
    std::cout << header;

    for (uint32_t i = 0; i < m_Model.m_EventClasses.size(); ++i)
    {
        std::string body;

        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model.m_EventClasses[i].m_Guid), 12));
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout << body;
    }
}

void TimelineDecoder::printEvents()
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

    for (uint32_t i = 0; i < m_Model.m_Events.size(); ++i)
    {
        std::string body;

        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model.m_Events[i].m_TimeStamp), 12));
        body.append(" | ");

        std::stringstream ss;
        ss << m_Model.m_Events[i].m_ThreadId;
        std::string threadId = ss.str();;

        body.append(profiling::CentreAlignFormatting(threadId, 12));
        body.append(" | ");
        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model.m_Events[i].m_Guid), 12));
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout << body;
    }
}

void TimelineDecoder::printRelationships()
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

    for (uint32_t i = 0; i < m_Model.m_Relationships.size(); ++i)
    {
        std::string body;

        body.append(
                profiling::CentreAlignFormatting(std::to_string(static_cast<unsigned int>
                                                                (m_Model.m_Relationships[i].m_RelationshipType)),
                                                 20));
        body.append(" | ");
        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model.m_Relationships[i].m_Guid), 20));
        body.append(" | ");
        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model.m_Relationships[i].m_HeadGuid), 12));
        body.append(" | ");
        body.append(profiling::CentreAlignFormatting(std::to_string(m_Model.m_Relationships[i].m_TailGuid), 12));
        body.append(" | ");
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout << body;
    }
}
}