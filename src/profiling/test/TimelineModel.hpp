//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <common/include/ProfilingException.hpp>
#include <server/include/timelineDecoder/ITimelineDecoder.hpp>

#include <map>
#include <sstream>
#include <vector>

namespace arm
{

namespace pipe
{
using LabelMap = std::map<uint64_t, arm::pipe::ITimelineDecoder::Label>;
using Attribute = std::pair<std::string, std::string>;
using Attributes = std::map<std::string, Attribute>;
class Entity;
class Connection
{
public:
    Connection(uint64_t guid, Entity* head, Entity* tail) :
        m_Guid(guid), m_HeadEntity(head), m_TailEntity(tail)
    {
        if (head == nullptr)
        {
            std::stringstream ss;
            ss << "connection [" << guid << "] head cannot be null";
            throw arm::pipe::ProfilingException(ss.str());
        }
        if (tail == nullptr)
        {
            std::stringstream ss;
            ss << "connection [" << guid << "] tail cannot be null";
            throw arm::pipe::ProfilingException(ss.str());
        }
    }

    uint64_t GetGuid() const {return m_Guid;}
    const Entity* GetHead() const {return m_HeadEntity;}
    const Entity* GetTail() const {return m_TailEntity;}
private:
    uint64_t m_Guid;
    Entity* m_HeadEntity;
    Entity* m_TailEntity;
};
class EventClassObj
{
public:
    EventClassObj(uint64_t guid, const std::string& name) : m_Guid(guid), m_Name(name) {}
    uint64_t GetGuid() const {return m_Guid;}
    const std::string& GetName() const {return m_Name;}
private:
    uint64_t m_Guid;
    std::string m_Name;
};
class EventObj
{
public:
    EventObj(uint64_t guid, uint64_t timestamp, uint64_t threadId) :
        m_Guid(guid), m_TimeStamp(timestamp), m_ThreadId(threadId) {}
    uint64_t GetGuid() const {return m_Guid;}
    uint64_t GetTimeStamp() const {return m_TimeStamp;}
    uint64_t GetThreadId() const {return m_ThreadId;}
    void SetEventClass(EventClassObj* evtClass) {m_EventClass = evtClass;}
    std::string GetEventClass()
    {
        if (m_EventClass == nullptr)
        {
            return "";
        }
        else
        {
            return m_EventClass->GetName();
        }
    }
private:
    uint64_t m_Guid;
    uint64_t m_TimeStamp;
    uint64_t m_ThreadId;
    EventClassObj* m_EventClass;
};
class Entity
{
public:
    Entity(uint64_t guid) : m_Guid(guid) {}
    uint64_t GetGuid() const {return m_Guid;}
    void AddChild(Entity* child)
    {
        if (child != nullptr)
        {
            m_Children.push_back(child);
        }
    }
    void AddAttribute(const std::string& type, const std::string& value)
    {
        Attribute attr(type, value);
        m_Attributes.emplace(type, attr);
    }
    void AddConnection(const Connection& connection)
    {
        m_Connections.push_back(connection);
    }
    void AddExecution(Entity* execution)
    {
        if (execution != nullptr)
        {
            m_Executions.push_back(execution);
        }
    }
    void AddEvent(EventObj* event)
    {
        if (event != nullptr)
        {
            m_Events.push_back(event);
        }
    }
    const Attributes& GetAttributes() const {return m_Attributes;}
    const std::vector<Entity*>& GetChildren() const {return m_Children;}
    const std::vector<Connection>& GetConnections() const {return m_Connections;}
    const std::vector<Entity*>& GetExecutions() const {return m_Executions;}
    const std::vector<EventObj*>& GetEvents() const {return m_Events;}
private:
    uint64_t m_Guid;
    Attributes m_Attributes;
    std::vector<Entity*> m_Children;
    std::vector<Connection> m_Connections;
    std::vector<Entity*> m_Executions;
    std::vector<EventObj*> m_Events;
};
using Entities = std::map<uint64_t, Entity>;
struct ModelRelationship
{
    ModelRelationship(const arm::pipe::ITimelineDecoder::Relationship& relationship) : m_Relationship(relationship) {}
    arm::pipe::ITimelineDecoder::Relationship m_Relationship;
    std::vector<Entity*> m_RelatedEntities;
};
using Relationships = std::map<uint64_t, ModelRelationship>;
using EventClasses = std::map<uint64_t, EventClassObj>;
using Events = std::map<uint64_t, EventObj>;
class TimelineModel
{
public:
    void AddLabel(const arm::pipe::ITimelineDecoder::Label& label);
    std::string* FindLabel(uint64_t guid);
    void AddEntity(uint64_t guid);
    Entity* FindEntity(uint64_t id);
    void AddRelationship(const arm::pipe::ITimelineDecoder::Relationship& relationship);
    ModelRelationship* FindRelationship(uint64_t id);
    const LabelMap& GetLabelMap() const {return m_LabelMap;}
    const Entities& GetEntities() const {return m_Entities;}
    const std::vector<arm::pipe::ProfilingException>& GetErrors() const {return m_Errors;}
    bool IsInferenceGuid(uint64_t guid) const;
    void AddEventClass(const arm::pipe::ITimelineDecoder::EventClass& eventClass);
    const EventClasses& GetEventClasses() const {return m_EventClasses;}
    EventClassObj* FindEventClass(uint64_t id);
    void AddEvent(const arm::pipe::ITimelineDecoder::Event& event);
    EventObj* FindEvent(uint64_t id);
private:
    LabelMap m_LabelMap;
    Entities m_Entities;
    Relationships m_Relationships;
    std::vector<arm::pipe::ProfilingException> m_Errors;
    std::vector<uint64_t> m_InferenceGuids;
    EventClasses m_EventClasses;
    Events m_Events;

    void HandleLabelLink(const arm::pipe::ITimelineDecoder::Relationship& relationship);
    void HandleConnection(const arm::pipe::ITimelineDecoder::Relationship& relationship);
    void HandleChild(const arm::pipe::ITimelineDecoder::Relationship& relationship);
    void HandleExecutionOf(const arm::pipe::ITimelineDecoder::Relationship& relationship);
    void HandleExecutionLink(const arm::pipe::ITimelineDecoder::Relationship& relationship);
};

std::vector<std::string> GetModelDescription(const TimelineModel& model);
std::string GetEntityDescription(const Entity& entity);
std::string GetChildDescription(Entity* entity);
std::string GetConnectionDescription(const Connection& connection);
std::string GetExecutionDescription(Entity* execution);
std::string GetEventDescription(EventObj* event);

} // namespace pipe

} // namespace arm