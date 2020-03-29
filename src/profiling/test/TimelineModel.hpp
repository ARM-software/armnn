//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/profiling/ITimelineDecoder.hpp>

#include <map>
#include <vector>

namespace armnn
{

namespace profiling
{
using LabelMap = std::map<uint64_t, ITimelineDecoder::Label>;
using Attribute = std::pair<std::string, std::string>;
using Attributes = std::map<std::string, Attribute>;
class Entity
{
public:
    Entity(uint64_t guid) : m_Guid(guid) {}
    uint64_t GetGuid() {return m_Guid;}
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
private:
    uint64_t m_Guid;
    Attributes m_Attributes;
    std::vector<Entity*> m_Children;
};
using Entities = std::map<uint64_t, Entity>;
struct ModelRelationship
{
    ModelRelationship(const ITimelineDecoder::Relationship& relationship) : m_Relationship(relationship) {}
    ITimelineDecoder::Relationship m_Relationship;
    std::vector<Entity*> m_RelatedEntities;
};
using Relationships = std::map<uint64_t, ModelRelationship>;
class TimelineModel
{
public:
    void AddLabel(const ITimelineDecoder::Label& label);
    void AddEntity(uint64_t guid);
    Entity* findEntity(uint64_t id);
    void AddRelationship(const ITimelineDecoder::Relationship& relationship);
    ModelRelationship* findRelationship(uint64_t id);
private:
    LabelMap m_LabelMap;
    Entities m_Entities;
    Relationships m_Relationships;
};

} // namespace profiling

} // namespace armnn