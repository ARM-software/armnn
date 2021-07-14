//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <server/include/timelineDecoder/ITimelineDecoder.hpp>

#include <armnnUtils/Filesystem.hpp>
#include <map>
#include <vector>

namespace armnn
{
namespace timelinedecoder
{
class JSONTimelineDecoder : public arm::pipe::ITimelineDecoder
{
public:
    struct JSONEntity
    {
    public:
        std::vector<uint64_t> connected_entities;
        std::vector<uint64_t> childEntities;

        JSONEntity(uint64_t guid): m_Guid(guid){}
        uint64_t GetGuid();
        std::string GetName();
        std::string GetType();
        void SetName(std::string entityName);
        void SetType(std::string entityType);
        void SetParent(JSONEntity& parent);
        void AddConnection(JSONEntity& headEntity, JSONEntity& connectedEntity);
        std::map<std::string, std::string> extendedData;

    private:
        uint64_t m_Guid;
        std::string name;
        std::string type;
    };

    struct Model
    {
        std::map<uint64_t, JSONEntity> jsonEntities;
        std::map<uint64_t, Relationship> relationships;
        std::map<uint64_t, Label> labels;
        std::map<uint64_t, Event> events;
        std::map<uint64_t, EventClass> eventClasses;
    };

    void PrintJSON(JSONEntity& entity, std::ostream& os);
    std::string GetJSONString(JSONEntity& rootEntity);
    std::string GetJSONEntityString(JSONEntity& entity, int& counter);

    virtual TimelineStatus CreateEntity(const Entity&) override;
    virtual TimelineStatus CreateEventClass(const EventClass&) override;
    virtual TimelineStatus CreateEvent(const Event&) override;
    virtual TimelineStatus CreateLabel(const Label&) override;
    virtual TimelineStatus CreateRelationship(const Relationship&) override;

    const Model& GetModel();

private:
    Model m_Model;

    void HandleRetentionLink(const Relationship& relationship);
    void HandleLabelLink(const Relationship& relationship);
    void HandleExecutionLink(const Relationship& relationship);
    void HandleConnectionLabel(const Relationship& relationship);
    void HandleBackendIdLabel(const Relationship& relationship);
    void HandleNameLabel(const Relationship& relationship);
    void HandleTypeLabel(const Relationship& relationship);

    std::string GetLayerJSONString(JSONEntity& entity, int& counter, std::string& jsonEntityString);
    std::string GetWorkloadJSONString(const JSONEntity& entity, int& counter, std::string& jsonEntityString);
    std::string GetWorkloadExecutionJSONString(const JSONEntity& entity, std::string& jsonEntityString) const;
};

}
}