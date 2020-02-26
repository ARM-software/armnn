//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "armnn/profiling/ITimelineDecoder.hpp"
#include <vector>

namespace armnn
{
namespace timelinedecoder
{
class TimelineDecoder : public ITimelineDecoder
{

public:

    struct Model
    {
        std::vector<Entity> m_Entities;
        std::vector<EventClass> m_EventClasses;
        std::vector<Event> m_Events;
        std::vector<Label> m_Labels;
        std::vector<Relationship> m_Relationships;
    };

    using OnNewEntityCallback       =  void (*)(Model &, const Entity);
    using OnNewEventClassCallback   =  void (*)(Model &, const EventClass);
    using OnNewEventCallback        =  void (*)(Model &, const Event);
    using OnNewLabelCallback        =  void (*)(Model &, const Label);
    using OnNewRelationshipCallback =  void (*)(Model &, const Relationship);

    virtual ErrorCode CreateEntity(const Entity &) override;
    virtual ErrorCode CreateEventClass(const EventClass &) override;
    virtual ErrorCode CreateEvent(const Event &) override;
    virtual ErrorCode CreateLabel(const Label &) override;
    virtual ErrorCode CreateRelationship(const Relationship &) override;

    const Model& GetModel();


    ErrorCode SetEntityCallback(const OnNewEntityCallback);
    ErrorCode SetEventClassCallback(const OnNewEventClassCallback);
    ErrorCode SetEventCallback(const OnNewEventCallback);
    ErrorCode SetLabelCallback(const OnNewLabelCallback);
    ErrorCode SetRelationshipCallback(const OnNewRelationshipCallback);

    void print();

private:
    Model m_Model;

    OnNewEntityCallback m_OnNewEntityCallback;
    OnNewEventClassCallback m_OnNewEventClassCallback;
    OnNewEventCallback m_OnNewEventCallback;
    OnNewLabelCallback m_OnNewLabelCallback;
    OnNewRelationshipCallback m_OnNewRelationshipCallback;

    void printLabels();
    void printEntities();
    void printEventClasses();
    void printRelationships();
    void printEvents();
};

}
}