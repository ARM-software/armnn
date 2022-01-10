//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "ITimelineDecoder.hpp"

#include <mutex>
#include <utility>
#include <vector>

namespace arm
{

namespace pipe
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

    virtual TimelineStatus CreateEntity(const Entity &) override;
    virtual TimelineStatus CreateEventClass(const EventClass &) override;
    virtual TimelineStatus CreateEvent(const Event &) override;
    virtual TimelineStatus CreateLabel(const Label &) override;
    virtual TimelineStatus CreateRelationship(const Relationship &) override;

    template<class F>
    decltype(auto) ApplyToModel(F&& f){
        std::lock_guard<std::mutex> lock(m_ModelMutex);
        return f(m_Model);
    }

    TimelineStatus SetEntityCallback(const OnNewEntityCallback);
    TimelineStatus SetEventClassCallback(const OnNewEventClassCallback);
    TimelineStatus SetEventCallback(const OnNewEventCallback);
    TimelineStatus SetLabelCallback(const OnNewLabelCallback);
    TimelineStatus SetRelationshipCallback(const OnNewRelationshipCallback);

    void SetDefaultCallbacks();

    void print();

private:
    Model m_Model;
    std::mutex m_ModelMutex;

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

} // namespace pipe
} // namespace arm
