//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TimelineModel.hpp"

#include <common/include/LabelsAndEventClasses.hpp>

#include <algorithm>

namespace arm
{

namespace pipe
{

void TimelineModel::AddLabel(const arm::pipe::ITimelineDecoder::Label& label)
{
    m_LabelMap.emplace(label.m_Guid, label);
}

std::string* TimelineModel::FindLabel(uint64_t guid)
{
    auto iter = m_LabelMap.find(guid);
    if (iter != m_LabelMap.end())
    {
        return &iter->second.m_Name;
    }
    else
    {
        return nullptr;
    }
}

void TimelineModel::AddEntity(uint64_t guid)
{
    m_Entities.emplace(guid, guid);
}

Entity* TimelineModel::FindEntity(uint64_t id)
{
    auto iter = m_Entities.find(id);
    if (iter != m_Entities.end())
    {
        return &(iter->second);
    }
    else
    {
        return nullptr;
    }
}

void TimelineModel::AddRelationship(const arm::pipe::ITimelineDecoder::Relationship& relationship)
{
    m_Relationships.emplace(relationship.m_Guid, relationship);
    if (relationship.m_RelationshipType == arm::pipe::ITimelineDecoder::RelationshipType::LabelLink)
    {
        HandleLabelLink(relationship);
    }
    else if (relationship.m_RelationshipType == ITimelineDecoder::RelationshipType::RetentionLink)
    {
        // Take care of the special case of a connection between layers in ArmNN
        // modelled by a retention link between two layer entities with an attribute GUID
        // of connection
        if (relationship.m_AttributeGuid == LabelsAndEventClasses::CONNECTION_GUID)
        {
            HandleConnection(relationship);
        }
        else if (relationship.m_AttributeGuid == LabelsAndEventClasses::CHILD_GUID)
        {
            HandleChild(relationship);
        }
        else if (relationship.m_AttributeGuid == LabelsAndEventClasses::EXECUTION_OF_GUID)
        {
            HandleExecutionOf(relationship);
        }
        else
        {
            // report unknown relationship type
            std::stringstream ss;
            ss << "Encountered a RetentionLink of unknown type [" << relationship.m_AttributeGuid << "]";
            m_Errors.push_back(arm::pipe::ProfilingException(ss.str()));
        }
    }
    else if (relationship.m_RelationshipType == arm::pipe::ITimelineDecoder::RelationshipType::ExecutionLink)
    {
        HandleExecutionLink(relationship);
    }
}

void TimelineModel::HandleLabelLink(const arm::pipe::ITimelineDecoder::Relationship& relationship)
{
    Entity* entity = FindEntity(relationship.m_HeadGuid);
    // we have a label attribute of an entity
    std::string* value = nullptr;
    std::string* attribute = nullptr;
    value = FindLabel(relationship.m_TailGuid);
    if (value == nullptr)
    {
        //report an error
        std::stringstream ss;
        ss << "could not find label link [" << relationship.m_Guid <<
           "] value [" << relationship.m_TailGuid << "]";
        m_Errors.push_back(arm::pipe::ProfilingException(ss.str()));
    }
    if (relationship.m_AttributeGuid != 0)
    {
        attribute = FindLabel(relationship.m_AttributeGuid);
        if (attribute == nullptr)
        {
            //report an error
            std::stringstream ss;
            ss << "could not find label link [" << relationship.m_Guid <<
               "] attribute [" << relationship.m_AttributeGuid << "]";
            m_Errors.push_back(arm::pipe::ProfilingException(ss.str()));
        }
    }
    else
    {
        //report an error
        std::stringstream ss;
        ss << "label link [" << relationship.m_Guid << "] has a zero attribute guid";
        m_Errors.push_back(arm::pipe::ProfilingException(ss.str()));
    }
    if (entity != nullptr && attribute != nullptr && value != nullptr)
    {
        entity->AddAttribute(*attribute, *value);
        // if the attribute is 'type' and the value is 'inference'
        // we need to cache the entity guid as an inference
        if (LabelsAndEventClasses::TYPE_LABEL.compare(*attribute) == 0 &&
            LabelsAndEventClasses::INFERENCE.compare(*value) == 0)
        {
            m_InferenceGuids.push_back(relationship.m_HeadGuid);
        }
    }

    if (entity == nullptr)
    {
        //report an error
        std::stringstream ss;
        ss << "could not find label link [" << relationship.m_Guid <<
           "] entity [" << relationship.m_HeadGuid << "] ";
        if (value != nullptr)
        {
            ss << "value [" << *value << "] ";
        }
        if (attribute != nullptr)
        {
            ss << "attribute [" << *attribute << "] ";
        }
        m_Errors.push_back(arm::pipe::ProfilingException(ss.str()));
    }
}

void TimelineModel::HandleConnection(const arm::pipe::ITimelineDecoder::Relationship& relationship)
{
    Entity* outputLayer = FindEntity(relationship.m_HeadGuid);
    if (outputLayer == nullptr)
    {
        std::stringstream ss;
        ss << "could not find output entity [" << relationship.m_HeadGuid << "]";
        ss << " of connection [" << relationship.m_Guid << "]";
        m_Errors.push_back(arm::pipe::ProfilingException(ss.str()));
        return;
    }
    Entity* inputLayer = FindEntity(relationship.m_TailGuid);
    if (inputLayer == nullptr)
    {
        std::stringstream ss;
        ss << "could not find input entity [" << relationship.m_TailGuid << "]";
        ss << " of connection [" << relationship.m_Guid << "]";
        m_Errors.push_back(arm::pipe::ProfilingException(ss.str()));
        return;
    }
    Connection connection(relationship.m_Guid, outputLayer, inputLayer);
    outputLayer->AddConnection(connection);
}

void TimelineModel::HandleChild(const arm::pipe::ITimelineDecoder::Relationship& relationship)
{
    Entity* parentEntity = FindEntity(relationship.m_HeadGuid);
    if (parentEntity == nullptr)
    {
        std::stringstream ss;
        ss << "could not find parent entity [" << relationship.m_HeadGuid << "]";
        ss << " of child relationship [" << relationship.m_Guid << "]";
        m_Errors.push_back(arm::pipe::ProfilingException(ss.str()));
        return;
    }
    Entity* childEntity = FindEntity(relationship.m_TailGuid);
    if (childEntity == nullptr)
    {
        std::stringstream ss;
        ss << "could not find child entity [" << relationship.m_TailGuid << "]";
        ss << " of child relationship [" << relationship.m_Guid << "]";
        m_Errors.push_back(arm::pipe::ProfilingException(ss.str()));
        return;
    }
    parentEntity->AddChild(childEntity);
}

void TimelineModel::HandleExecutionOf(const arm::pipe::ITimelineDecoder::Relationship& relationship)
{
    Entity* parentEntity = FindEntity(relationship.m_HeadGuid);
    if (parentEntity == nullptr)
    {
        std::stringstream ss;
        ss << "could not find parent entity [" << relationship.m_HeadGuid << "]";
        ss << " of execution relationship [" << relationship.m_Guid << "]";
        m_Errors.push_back(arm::pipe::ProfilingException(ss.str()));
        return;
    }
    Entity* executedEntity = FindEntity(relationship.m_TailGuid);
    if (executedEntity == nullptr)
    {
        std::stringstream ss;
        ss << "could not find executed entity [" << relationship.m_TailGuid << "]";
        ss << " of execution relationship [" << relationship.m_Guid << "]";
        m_Errors.push_back(arm::pipe::ProfilingException(ss.str()));
        return;
    }
    parentEntity->AddExecution(executedEntity);
}

void TimelineModel::HandleExecutionLink(const arm::pipe::ITimelineDecoder::Relationship& relationship)
{
    // entityGuid,
    Entity* parentEntity = FindEntity(relationship.m_HeadGuid);
    if (parentEntity == nullptr)
    {
        std::stringstream ss;
        ss << "could not find entity [" << relationship.m_HeadGuid << "]";
        ss << " of ExecutionLink [" << relationship.m_Guid << "]";
        m_Errors.push_back(arm::pipe::ProfilingException(ss.str()));
        return;
    }
    // eventGuid,
    EventObj* eventObj = FindEvent(relationship.m_TailGuid);
    if (eventObj == nullptr)
    {
        std::stringstream ss;
        ss << "could not find event [" << relationship.m_TailGuid << "]";
        ss << " of ExecutionLink [" << relationship.m_Guid << "]";
        m_Errors.push_back(arm::pipe::ProfilingException(ss.str()));
        return;
    }
    // eventClassGuid
    EventClassObj* eventClassObj = FindEventClass(relationship.m_AttributeGuid);
    if (eventClassObj == nullptr)
    {
        std::stringstream ss;
        ss << "could not find event class [" << relationship.m_TailGuid << "]";
        ss << " of ExecutionLink [" << relationship.m_Guid << "]";
        m_Errors.push_back(arm::pipe::ProfilingException(ss.str()));
        return;
    }
    eventObj->SetEventClass(eventClassObj);
    parentEntity->AddEvent(eventObj);
}

ModelRelationship* TimelineModel::FindRelationship(uint64_t id)
{
    auto iter = m_Relationships.find(id);
    if (iter != m_Relationships.end())
    {
        return &(iter->second);
    }
    else
    {
        return nullptr;
    }
}

bool TimelineModel::IsInferenceGuid(uint64_t guid) const
{
    auto it = std::find(m_InferenceGuids.begin(), m_InferenceGuids.end(), guid);
    return it != m_InferenceGuids.end();
}

void TimelineModel::AddEventClass(const arm::pipe::ITimelineDecoder::EventClass& eventClass)
{
    std::string* eventClassName = FindLabel(eventClass.m_NameGuid);
    if (eventClassName != nullptr)
    {
        EventClassObj eventClassObj(eventClass.m_Guid, *eventClassName);
        m_EventClasses.emplace(eventClassObj.GetGuid(), eventClassObj);
    }
    else
    {
        std::stringstream ss;
        ss << "could not find name [" << eventClass.m_NameGuid << "]";
        ss << " of of event class  [" << eventClass.m_Guid << "]";
        m_Errors.push_back(arm::pipe::ProfilingException(ss.str()));
    }
}

EventClassObj* TimelineModel::FindEventClass(uint64_t id)
{
    auto iter = m_EventClasses.find(id);
    if (iter != m_EventClasses.end())
    {
        return &(iter->second);
    }
    else
    {
        return nullptr;
    }
}

void TimelineModel::AddEvent(const arm::pipe::ITimelineDecoder::Event& event)
{
    EventObj evt(event.m_Guid, event.m_TimeStamp, event.m_ThreadId);
    m_Events.emplace(event.m_Guid, evt);
}

EventObj* TimelineModel::FindEvent(uint64_t id)
{
    auto iter = m_Events.find(id);
    if (iter != m_Events.end())
    {
        return &(iter->second);
    }
    else
    {
        return nullptr;
    }
}

std::vector<std::string> GetModelDescription(const TimelineModel& model)
{
    std::vector<std::string> desc;
    for (auto& entry : model.GetEntities())
    {
        auto& entity = entry.second;
        desc.push_back(GetEntityDescription(entity));
        for (auto& connection : entity.GetConnections())
        {
            desc.push_back(GetConnectionDescription(connection));
        }
        for (auto child : entity.GetChildren())
        {
            desc.push_back(GetChildDescription(child));
        }
        for (auto execution : entity.GetExecutions())
        {
            desc.push_back(GetExecutionDescription(execution));
        }
        for (auto event : entity.GetEvents())
        {
            desc.push_back(GetEventDescription(event));
        }
    }
    return desc;
}

std::string GetEntityDescription(const Entity& entity)
{
    std::stringstream ss;
    ss << "Entity [" << entity.GetGuid() << "]";
    for (auto& attributeEntry : entity.GetAttributes())
    {
        if (LabelsAndEventClasses::PROCESS_ID_LABEL == attributeEntry.second.first)
        {
            ss << " " << attributeEntry.second.first << " = [processId]";
        }
        else {
            ss << " " << attributeEntry.second.first << " = " << attributeEntry.second.second;
        }
    }
    return ss.str();
}

std::string GetChildDescription(Entity* entity)
{
    std::stringstream ss;
    ss << "   child: " << GetEntityDescription(*entity);
    return ss.str();
}

std::string GetConnectionDescription(const Connection& connection)
{
    std::stringstream ss;
    ss << "   connection [" << connection.GetGuid() << "] from entity [";
    ss << connection.GetHead()->GetGuid() << "] to entity [" << connection.GetTail()->GetGuid() << "]";
    return ss.str();
}

std::string GetExecutionDescription(Entity* execution)
{
    std::stringstream ss;
    ss << "   execution: " << GetEntityDescription(*execution);
    return ss.str();
}

std::string GetEventDescription(EventObj* event)
{
    std::stringstream ss;
    ss << "   event: [" << event->GetGuid() << "] class [" << event->GetEventClass() << "]";
    return ss.str();
}

} // namespace pipe

} // namespace arm