//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <common/include/CommonProfilingUtils.hpp>
#include <server/include/timelineDecoder/TimelineDecoder.hpp>

#include <iostream>
#include <sstream>

namespace arm
{
namespace pipe
{

TimelineDecoder::TimelineStatus TimelineDecoder::CreateEntity(const Entity &entity)
{
    if (m_OnNewEntityCallback == nullptr)
    {
        return TimelineStatus::TimelineStatus_Fail;
    }
    ApplyToModel([&](Model& m){
        m_OnNewEntityCallback(m, entity);
    });
    return TimelineStatus::TimelineStatus_Success;
}

TimelineDecoder::TimelineStatus TimelineDecoder::CreateEventClass(const EventClass &eventClass)
{
    if (m_OnNewEventClassCallback == nullptr)
    {
        return TimelineStatus::TimelineStatus_Fail;
    }
    ApplyToModel([&](Model& m){
        m_OnNewEventClassCallback(m, eventClass);
    });

    return TimelineStatus::TimelineStatus_Success;
}

TimelineDecoder::TimelineStatus TimelineDecoder::CreateEvent(const Event &event)
{
    if (m_OnNewEventCallback == nullptr)
    {
        return TimelineStatus::TimelineStatus_Fail;
    }
    ApplyToModel([&](Model& m){
        m_OnNewEventCallback(m, event);
    });

    return TimelineStatus::TimelineStatus_Success;
}

TimelineDecoder::TimelineStatus TimelineDecoder::CreateLabel(const Label &label)
{
    if (m_OnNewLabelCallback == nullptr)
    {
        return TimelineStatus::TimelineStatus_Fail;
    }
    ApplyToModel([&](Model& m){
        m_OnNewLabelCallback(m, label);
    });

    return TimelineStatus::TimelineStatus_Success;
}

TimelineDecoder::TimelineStatus TimelineDecoder::CreateRelationship(const Relationship &relationship)
{
    if (m_OnNewRelationshipCallback == nullptr)
    {
        return TimelineStatus::TimelineStatus_Fail;
    }
    ApplyToModel([&](Model& m){
        m_OnNewRelationshipCallback(m, relationship);
    });
    return TimelineStatus::TimelineStatus_Success;
}

TimelineDecoder::TimelineStatus TimelineDecoder::SetEntityCallback(OnNewEntityCallback cb)
{
    if (cb == nullptr)
    {
        return TimelineStatus::TimelineStatus_Fail;
    }
    m_OnNewEntityCallback = cb;
    return TimelineStatus::TimelineStatus_Success;
}

TimelineDecoder::TimelineStatus TimelineDecoder::SetEventClassCallback(OnNewEventClassCallback cb)
{
    if (cb == nullptr)
    {
        return TimelineStatus::TimelineStatus_Fail;
    }
    m_OnNewEventClassCallback = cb;
    return TimelineStatus::TimelineStatus_Success;
}

TimelineDecoder::TimelineStatus TimelineDecoder::SetEventCallback(OnNewEventCallback cb)
{
    if (cb == nullptr)
    {
        return TimelineStatus::TimelineStatus_Fail;
    }
    m_OnNewEventCallback = cb;
    return TimelineStatus::TimelineStatus_Success;
}

TimelineDecoder::TimelineStatus TimelineDecoder::SetLabelCallback(OnNewLabelCallback cb)
{
    if (cb == nullptr)
    {
        return TimelineStatus::TimelineStatus_Fail;
    }
    m_OnNewLabelCallback = cb;
    return TimelineStatus::TimelineStatus_Success;
}

TimelineDecoder::TimelineStatus TimelineDecoder::SetRelationshipCallback(OnNewRelationshipCallback cb)
{
    if (cb == nullptr)
    {
        return TimelineStatus::TimelineStatus_Fail;
    }
    m_OnNewRelationshipCallback = cb;
    return TimelineStatus::TimelineStatus_Success;
}

void TimelineDecoder::SetDefaultCallbacks()
{
    SetEntityCallback([](Model& model, const ITimelineDecoder::Entity entity)
    {
        model.m_Entities.emplace_back(entity);
    });

    SetEventClassCallback([](Model& model, const ITimelineDecoder::EventClass eventClass)
    {
        model.m_EventClasses.emplace_back(eventClass);
    });

    SetEventCallback([](Model& model, const ITimelineDecoder::Event event)
    {
        model.m_Events.emplace_back(event);
    });

    SetLabelCallback([](Model& model, const ITimelineDecoder::Label label)
    {
        model.m_Labels.emplace_back(label);
    });

    SetRelationshipCallback([](Model& model, const ITimelineDecoder::Relationship relationship)
    {
        model.m_Relationships.emplace_back(relationship);
    });
}

void TimelineDecoder::print()
{
    if (m_Model.m_Labels.empty() && m_Model.m_Entities.empty() && m_Model.m_EventClasses.empty() &&
        m_Model.m_Events.empty() && m_Model.m_Relationships.empty())
    {
        std::cout << "No timeline packets received" << std::endl;
        return;
    }

    printLabels();
    printEntities();
    printEventClasses();
    printEvents();
    printRelationships();
}

void TimelineDecoder::printLabels()
{
    std::string header;

    header.append(arm::pipe::CentreAlignFormatting("guid", 12));
    header.append(" | ");
    header.append(arm::pipe::CentreAlignFormatting("value", 30));
    header.append("\n");

    std::cout << "\n" << "\n";
    std::cout << arm::pipe::CentreAlignFormatting("LABELS", static_cast<int>(header.size()));
    std::cout << "\n";
    std::cout << std::string(header.size(), '=') << "\n";
    std::cout << header;

    for (uint32_t i = 0; i < m_Model.m_Labels.size(); ++i)
    {
        std::string body;

        body.append(arm::pipe::CentreAlignFormatting(std::to_string(m_Model.m_Labels[i].m_Guid), 12));
        body.append(" | ");
        body.append(arm::pipe::CentreAlignFormatting(m_Model.m_Labels[i].m_Name, 30));
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout << body;
    }
}

void TimelineDecoder::printEntities()
{
    std::string header;
    header.append(arm::pipe::CentreAlignFormatting("guid", 12));
    header.append("\n");

    std::cout << "\n" << "\n";
    std::cout << arm::pipe::CentreAlignFormatting("ENTITIES", static_cast<int>(header.size()));
    std::cout << "\n";
    std::cout << std::string(header.size(), '=') << "\n";
    std::cout << header;

    for (uint32_t i = 0; i < m_Model.m_Entities.size(); ++i)
    {
        std::string body;

        body.append(arm::pipe::CentreAlignFormatting(std::to_string(m_Model.m_Entities[i].m_Guid), 12));
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout << body;
    }
}

void TimelineDecoder::printEventClasses()
{
    std::string header;
    header.append(arm::pipe::CentreAlignFormatting("guid", 12));
    header.append("\n");

    std::cout << "\n" << "\n";
    std::cout << arm::pipe::CentreAlignFormatting("EVENT CLASSES", static_cast<int>(header.size()));
    std::cout << "\n";
    std::cout << std::string(header.size(), '=') << "\n";
    std::cout << header;

    for (uint32_t i = 0; i < m_Model.m_EventClasses.size(); ++i)
    {
        std::string body;

        body.append(arm::pipe::CentreAlignFormatting(std::to_string(m_Model.m_EventClasses[i].m_Guid), 12));
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout << body;
    }
}

void TimelineDecoder::printEvents()
{
    std::string header;

    header.append(arm::pipe::CentreAlignFormatting("timestamp", 12));
    header.append(" | ");
    header.append(arm::pipe::CentreAlignFormatting("threadId", 12));
    header.append(" | ");
    header.append(arm::pipe::CentreAlignFormatting("eventGuid", 12));
    header.append("\n");

    std::cout << "\n" << "\n";
    std::cout << arm::pipe::CentreAlignFormatting("EVENTS", static_cast<int>(header.size()));
    std::cout << "\n";
    std::cout << std::string(header.size(), '=') << "\n";
    std::cout << header;

    for (uint32_t i = 0; i < m_Model.m_Events.size(); ++i)
    {
        std::string body;

        body.append(arm::pipe::CentreAlignFormatting(std::to_string(m_Model.m_Events[i].m_TimeStamp), 12));
        body.append(" | ");

        std::stringstream ss;
        ss << m_Model.m_Events[i].m_ThreadId;
        std::string threadId = ss.str();;

        body.append(arm::pipe::CentreAlignFormatting(threadId, 12));
        body.append(" | ");
        body.append(arm::pipe::CentreAlignFormatting(std::to_string(m_Model.m_Events[i].m_Guid), 12));
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout << body;
    }
}

void TimelineDecoder::printRelationships()
{
    std::string header;
    header.append(arm::pipe::CentreAlignFormatting("relationshipType", 20));
    header.append(" | ");
    header.append(arm::pipe::CentreAlignFormatting("relationshipGuid", 20));
    header.append(" | ");
    header.append(arm::pipe::CentreAlignFormatting("headGuid", 12));
    header.append(" | ");
    header.append(arm::pipe::CentreAlignFormatting("tailGuid", 12));
    header.append("\n");

    std::cout << "\n" << "\n";
    std::cout << arm::pipe::CentreAlignFormatting("RELATIONSHIPS", static_cast<int>(header.size()));
    std::cout << "\n";
    std::cout << std::string(header.size(), '=') << "\n";
    std::cout << header;

    for (uint32_t i = 0; i < m_Model.m_Relationships.size(); ++i)
    {
        std::string body;

        body.append(
                arm::pipe::CentreAlignFormatting(std::to_string(static_cast<unsigned int>
                                                                (m_Model.m_Relationships[i].m_RelationshipType)),
                                                 20));
        body.append(" | ");
        body.append(arm::pipe::CentreAlignFormatting(std::to_string(m_Model.m_Relationships[i].m_Guid), 20));
        body.append(" | ");
        body.append(arm::pipe::CentreAlignFormatting(std::to_string(m_Model.m_Relationships[i].m_HeadGuid), 12));
        body.append(" | ");
        body.append(arm::pipe::CentreAlignFormatting(std::to_string(m_Model.m_Relationships[i].m_TailGuid), 12));
        body.append(" | ");
        body.append("\n");

        std::cout << std::string(body.size(), '-') << "\n";
        std::cout << body;
    }
}

} // namespace pipe
} // namespace arm
