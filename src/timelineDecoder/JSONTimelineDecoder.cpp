//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "JSONTimelineDecoder.hpp"

#include <client/src/ProfilingUtils.hpp>

#include <string>

namespace armnn
{
namespace timelinedecoder
{

static const char *const CONNECTION = "connection";
static const char *const BACKEND_ID = "backendId";
static const char *const NAME = "name";
static const char *const TYPE = "type";
static const char *const WORKLOAD = "workload";
static const char *const WORKLOAD_EXECUTION = "workload_execution";
static const char *const INFERENCE = "inference";
static const char *const LAYER = "layer";
static const char *const ENTITY = "Entity";
static const char *const EVENTCLASS = "EventClass";
static const char *const EVENT = "Event";

JSONTimelineDecoder::TimelineStatus JSONTimelineDecoder::CreateEntity(const Entity& entity)
{
    JSONEntity jsonEntity(entity.m_Guid);
    jsonEntity.SetType(ENTITY);
    this->m_Model.jsonEntities.insert({entity.m_Guid, jsonEntity});
    return TimelineStatus::TimelineStatus_Success;
}

JSONTimelineDecoder::TimelineStatus JSONTimelineDecoder::CreateEventClass(const EventClass& eventClass)
{
    JSONEntity jsonEntity(eventClass.m_Guid);
    jsonEntity.SetType(EVENTCLASS);
    this->m_Model.eventClasses.insert({eventClass.m_Guid, eventClass});
    this->m_Model.jsonEntities.insert({eventClass.m_Guid, jsonEntity});
    return TimelineStatus::TimelineStatus_Success;
}

JSONTimelineDecoder::TimelineStatus JSONTimelineDecoder::CreateEvent(const Event& event)
{
    JSONEntity jsonEntity(event.m_Guid);
    jsonEntity.SetType(EVENT);
    this->m_Model.events.insert({event.m_Guid, event});
    this->m_Model.jsonEntities.insert({jsonEntity.GetGuid(), jsonEntity});
    return TimelineStatus::TimelineStatus_Success;
}

JSONTimelineDecoder::TimelineStatus JSONTimelineDecoder::CreateLabel(const Label& label)
{
    this->m_Model.labels.insert({label.m_Guid, label});
    return TimelineStatus::TimelineStatus_Success;
}

JSONTimelineDecoder::TimelineStatus JSONTimelineDecoder::CreateRelationship(const Relationship& relationship)
{
    if (relationship.m_RelationshipType == ITimelineDecoder::RelationshipType::RetentionLink)
    {
        HandleRetentionLink(relationship);
    }
    else if (relationship.m_RelationshipType == ITimelineDecoder::RelationshipType::LabelLink)
    {
        HandleLabelLink(relationship);
    }
    else if (relationship.m_RelationshipType == ITimelineDecoder::RelationshipType::ExecutionLink)
    {
        HandleExecutionLink(relationship);
    }
    else
    {
        /*
         * TODO Handle DataLink
         */
        m_Model.relationships.insert({relationship.m_Guid, relationship});
    }

    return TimelineStatus::TimelineStatus_Success;
}


void JSONTimelineDecoder::HandleExecutionLink(const ITimelineDecoder::Relationship& relationship)
{
    uint64_t tailGuid = relationship.m_TailGuid;
    uint64_t headGuid = relationship.m_HeadGuid;

    if (m_Model.jsonEntities.count(relationship.m_HeadGuid) != 0)
    {
        JSONEntity& tailJSONEntity = m_Model.jsonEntities.at(tailGuid);
        JSONEntity& headJSONEntity = m_Model.jsonEntities.at(headGuid);
        tailJSONEntity.SetParent(headJSONEntity);
        m_Model.jsonEntities.insert({headGuid, headJSONEntity});
        m_Model.relationships.insert({relationship.m_Guid, relationship});
    }
    else
    {
        /*
         * TODO Add some protection against packet ordering issues
         */
        m_Model.relationships.insert({relationship.m_Guid, relationship});
    }
}

void JSONTimelineDecoder::HandleLabelLink(const ITimelineDecoder::Relationship& relationship)
{
    if (m_Model.labels.count(relationship.m_TailGuid) != 0)
    {
        if (m_Model.labels.at(relationship.m_TailGuid).m_Name == CONNECTION)
        {
            HandleConnectionLabel(relationship);
        }
        else if (m_Model.labels.at(relationship.m_TailGuid).m_Name == BACKEND_ID)
        {
            HandleBackendIdLabel(relationship);
        }
        else if (m_Model.labels.at(relationship.m_TailGuid).m_Name == NAME)
        {
            HandleNameLabel(relationship);
        }
        else if (m_Model.labels.at(relationship.m_TailGuid).m_Name == TYPE)
        {
            HandleTypeLabel(relationship);
        }
        else
        {
            /*
             * TODO Add some protection against packet ordering issues
             */
            m_Model.relationships.insert({relationship.m_Guid, relationship});
        }
    } else
    {
        /*
         * TODO Add some protection against packet ordering issues
         */
        m_Model.relationships.insert({relationship.m_Guid, relationship});
    }
}

void JSONTimelineDecoder::HandleTypeLabel(const ITimelineDecoder::Relationship& relationship)
{
    if (m_Model.relationships.count(relationship.m_HeadGuid) != 0)
    {
        Relationship labelRelation = m_Model.relationships.at(relationship.m_HeadGuid);
        if (m_Model.jsonEntities.count(labelRelation.m_HeadGuid) != 0)
        {
            JSONEntity& headEntity = m_Model.jsonEntities.at(labelRelation.m_HeadGuid);
            std::string type = m_Model.labels.at(labelRelation.m_TailGuid).m_Name;
            headEntity.SetType(type);
        }
    }
    else
    {
        /*
        * TODO Add some protection against packet ordering issues
        */
        m_Model.relationships.insert({relationship.m_Guid, relationship});
    }
}

void JSONTimelineDecoder::HandleNameLabel(const ITimelineDecoder::Relationship& relationship)
{
    if (m_Model.relationships.count(relationship.m_HeadGuid) != 0)
    {
        Relationship labelRelation = m_Model.relationships.at(relationship.m_HeadGuid);
        JSONEntity& headEntity = m_Model.jsonEntities.at(labelRelation.m_HeadGuid);
        std::string name = m_Model.labels.at(labelRelation.m_TailGuid).m_Name;
        headEntity.SetName(name);
    }
    else
    {
        /*
        * TODO Add some protection against packet ordering issues
        */
        m_Model.relationships.insert({relationship.m_Guid, relationship});
    }
}

void JSONTimelineDecoder::HandleBackendIdLabel(const ITimelineDecoder::Relationship& relationship)
{
    if (m_Model.relationships.count(relationship.m_HeadGuid) != 0)
    {
        Relationship labelRelation = m_Model.relationships.at(relationship.m_HeadGuid);
        JSONEntity& headEntity = m_Model.jsonEntities.at(labelRelation.m_HeadGuid);
        std::string backendName = m_Model.labels.at(labelRelation.m_TailGuid).m_Name;
        headEntity.extendedData.insert({BACKEND_ID, backendName});
    }
    else
    {
        /*
        * TODO Add some protection against packet ordering issues
        */
        m_Model.relationships.insert({relationship.m_Guid, relationship});
    }
}

void JSONTimelineDecoder::HandleConnectionLabel(const ITimelineDecoder::Relationship& relationship)
{
    if (m_Model.relationships.count(relationship.m_HeadGuid) != 0)
    {
        Relationship retentionRelation = m_Model.relationships.at(relationship.m_HeadGuid);
        JSONEntity& headEntity = m_Model.jsonEntities.at(retentionRelation.m_HeadGuid);
        JSONEntity& tailEntity = m_Model.jsonEntities.at(retentionRelation.m_TailGuid);
        headEntity.AddConnection(headEntity, tailEntity);
    }
    else
    {
        /*
        * TODO Add some protection against packet ordering issues
        */
        m_Model.relationships.insert({relationship.m_Guid, relationship});
    }
}

void JSONTimelineDecoder::HandleRetentionLink(const ITimelineDecoder::Relationship& relationship)
{
    if (m_Model.jsonEntities.count(relationship.m_TailGuid) != 0 && m_Model.jsonEntities
    .count(relationship.m_HeadGuid) != 0)
    {
        JSONEntity& tailJSONEntity = m_Model.jsonEntities.at(relationship.m_TailGuid);
        JSONEntity& headJSONEntity = m_Model.jsonEntities.at(relationship.m_HeadGuid);
        tailJSONEntity.SetParent(headJSONEntity);
        m_Model.jsonEntities.insert({relationship.m_HeadGuid, headJSONEntity});
        m_Model.relationships.insert({relationship.m_Guid, relationship});
    }
    else
    {
        /*
        * TODO Add some protection against packet ordering issues
        */
        m_Model.relationships.insert({relationship.m_Guid, relationship});
    }
}

void JSONTimelineDecoder::JSONEntity::SetParent(JSONEntity& parent)
{
    parent.childEntities.push_back(GetGuid());
}

void JSONTimelineDecoder::PrintJSON(JSONTimelineDecoder::JSONEntity& rootEntity, std::ostream& os)
{
    std::string jsonString = GetJSONString(rootEntity);
    os << jsonString;
}

std::string JSONTimelineDecoder::GetJSONString(JSONTimelineDecoder::JSONEntity& rootEntity)
{
    int counter = 0;
    std::string json;
    json.append("{\n");
    if(rootEntity.GetType() != "")
    {
        json.append("\tArmNN");
        json.append(": {\n");

        for (uint64_t childEntityId : rootEntity.childEntities)
        {
            JSONEntity& childEntity = this->m_Model.jsonEntities.at(childEntityId);
            json.append(GetJSONEntityString(childEntity, counter));
        }
    }
    json.append("}\n");
    return json;
}

std::string JSONTimelineDecoder::GetJSONEntityString(JSONTimelineDecoder::JSONEntity& entity, int& counter)
{
    std::string jsonEntityString;
    if(entity.GetType() == LAYER)
    {
        return GetLayerJSONString(entity, counter, jsonEntityString);
    }
    else if (entity.GetType() == WORKLOAD)
    {
        return GetWorkloadJSONString(entity, counter, jsonEntityString);
    }
    else if (entity.GetType() == WORKLOAD_EXECUTION)
    {
        return GetWorkloadExecutionJSONString(entity, jsonEntityString);
    }
    else if (entity.GetType() == INFERENCE)
    {
        return jsonEntityString;
    }
    else
    {
        for (uint64_t child_entity_id : entity.childEntities)
        {
            JSONEntity& childEntity = this->m_Model.jsonEntities.at(child_entity_id);
            jsonEntityString.append(GetJSONEntityString(childEntity, ++counter));
        }
        return jsonEntityString;
    }
}

std::string JSONTimelineDecoder::GetWorkloadExecutionJSONString(const JSONTimelineDecoder::JSONEntity& entity,
                                                                std::string& jsonEntityString) const
{
    if(entity.childEntities.size() < 2)
    {
        throw arm::pipe::ProfilingException(
            "Workload Execution Entity Packet does not have the expected Event packets attached");
    }
    JSONEntity jsonEventOne = entity.childEntities[0];
    JSONEntity jsonEventTwo = entity.childEntities[1];

    Event event1 = m_Model.events.at(jsonEventOne.GetGuid());
    Event event2 = m_Model.events.at(jsonEventTwo.GetGuid());

    uint64_t wall_clock_time = event2.m_TimeStamp - event1.m_TimeStamp;
    jsonEntityString.append("\t\t\t");
    jsonEntityString.append("raw : [");
    jsonEntityString.append(std::to_string(wall_clock_time));
    jsonEntityString.append("], \n");
    jsonEntityString.append("\t\t\t");
    jsonEntityString.append("unit : us,\n");
    jsonEntityString.append("\t\t\t");
    jsonEntityString.append("}\n");

    return jsonEntityString;
}

std::string JSONTimelineDecoder::GetWorkloadJSONString(const JSONTimelineDecoder::JSONEntity& entity, int& counter,
                                                       std::string& jsonEntityString)
{
    jsonEntityString.append("\t\t\t");
    jsonEntityString.append("backendId :");
    jsonEntityString.append(entity.extendedData.at(BACKEND_ID));
    jsonEntityString.append(",\n");
    for (uint64_t child_entity_id : entity.childEntities)
    {
        JSONEntity &childEntity = m_Model.jsonEntities.at(child_entity_id);
        jsonEntityString.append(GetJSONEntityString(childEntity, ++counter));
    }
    return jsonEntityString;
}

std::string JSONTimelineDecoder::GetLayerJSONString(JSONTimelineDecoder::JSONEntity& entity, int& counter,
                                                    std::string& jsonEntityString)
{
    jsonEntityString.append("\t\t");
    jsonEntityString.append(entity.GetName());
    jsonEntityString.append("_");
    jsonEntityString.append(std::to_string(counter));
    jsonEntityString.append(": {\n");
    jsonEntityString.append("\t\t\t");
    jsonEntityString.append("type: Measurement,\n");
    for (uint64_t child_entity_id : entity.childEntities)
    {
        JSONEntity& childEntity = m_Model.jsonEntities.at(child_entity_id);
        jsonEntityString.append(GetJSONEntityString(childEntity, ++counter));
    }
    return jsonEntityString;
}

void JSONTimelineDecoder::JSONEntity::AddConnection(JSONEntity& headEntity, JSONEntity& connectedEntity)
{
    std::vector<uint64_t>::iterator it = std::find(headEntity.childEntities.begin(),
            headEntity.childEntities.end(), connectedEntity.GetGuid());
    headEntity.childEntities.erase(it);
    headEntity.connected_entities.push_back(connectedEntity.m_Guid);
}

uint64_t JSONTimelineDecoder::JSONEntity::GetGuid()
{
    return m_Guid;
}

const JSONTimelineDecoder::Model &JSONTimelineDecoder::GetModel()
{
    return m_Model;
}

void JSONTimelineDecoder::JSONEntity::SetName(std::string entityName)
{
    this->name = entityName;
}

std::string JSONTimelineDecoder::JSONEntity::GetName()
{
    return this->name;
}

void JSONTimelineDecoder::JSONEntity::SetType(std::string entityType)
{
    this->type = entityType;
}

std::string JSONTimelineDecoder::JSONEntity::GetType()
{
    return this->type;
}

}
}
