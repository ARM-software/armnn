//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <algorithm>
#include "../TimelineModel.h"

void PushEntity(const Entity entity, Model* model)
{
    if(model->m_EntityCount == 0)
    {
        model->m_EntityCapacity = 1;
        model->m_Entities = new Entity*[model->m_EntityCapacity];
    }
    else if(model->m_EntityCount >= model->m_EntityCapacity)
    {
        Entity** newEntityArray = new Entity*[model->m_EntityCapacity*2];

        std::copy(model->m_Entities, model->m_Entities + model->m_EntityCapacity, newEntityArray);
        delete[] model->m_Entities;
        model->m_Entities = newEntityArray;

        model->m_EntityCapacity = model->m_EntityCapacity *2;
    }

    Entity* newEntity = new Entity;

    newEntity->m_Guid = entity.m_Guid;

    model->m_Entities[model->m_EntityCount] = newEntity;
    model->m_EntityCount++;
};

void PushEventClass(const EventClass eventClass, Model* model)
{
    if(model->m_EventClassCount == 0)
    {
        model->m_EventClassCapacity = 1;
        model->m_EventClasses = new EventClass*[model->m_EventClassCapacity];
    }
    else if(model->m_EventClassCount >= model->m_EventClassCapacity)
    {
        EventClass** newEventClassArray = new EventClass*[model->m_EventClassCapacity *2];

        std::copy(model->m_EventClasses, model->m_EventClasses + model->m_EventClassCapacity, newEventClassArray);
        delete[] model->m_EventClasses;
        model->m_EventClasses = newEventClassArray;

        model->m_EventClassCapacity = model->m_EventClassCapacity *2;
    }

    EventClass* newEventClass = new EventClass;

    newEventClass->m_Guid = eventClass.m_Guid;

    model->m_EventClasses[model->m_EventClassCount] = newEventClass;
    model->m_EventClassCount++;
};

void PushEvent(const Event event, Model* model)
{
    if(model->m_EventCount == 0)
    {
        model->m_EventCapacity = 1;
        model->m_Events = new Event*[model->m_EventCapacity];
    }
    else if(model->m_EventCount >= model->m_EventCapacity)
    {
        Event** newEventArray = new Event*[model->m_EventCapacity * 2];

        std::copy(model->m_Events, model->m_Events + model->m_EventCapacity, newEventArray);
        delete[] model->m_Events;
        model->m_Events = newEventArray;

        model->m_EventCapacity = model->m_EventCapacity *2;
    }

    Event* newEvent = new Event;

    newEvent->m_TimeStamp = event.m_TimeStamp;
    newEvent->m_ThreadId = event.m_ThreadId;
    newEvent->m_Guid = event.m_Guid;

    model->m_Events[model->m_EventCount] = newEvent;
    model->m_EventCount++;
};

void PushLabel(const Label label, Model* model)
{
    if(model->m_LabelCount == 0)
    {
        model->m_LabelCapacity = 1;
        model->m_Labels = new Label*[model->m_LabelCapacity];
    }
    else if(model->m_LabelCount >= model->m_LabelCapacity)
    {
        Label** newLabelArray = new Label*[model->m_LabelCapacity *2];

        std::copy(model->m_Labels, model->m_Labels + model->m_LabelCapacity, newLabelArray);
        delete[] model->m_Labels;
        model->m_Labels = newLabelArray;

        model->m_LabelCapacity = model->m_LabelCapacity *2;
    }

    Label* newLabel = new Label;

    newLabel->m_Guid = label.m_Guid;
    newLabel->m_Name = label.m_Name;

    model->m_Labels[model->m_LabelCount] = newLabel;
    model->m_LabelCount++;
};

void PushRelationship(const Relationship relationship, Model* model)
{
    if(model->m_RelationshipCount == 0)
    {
        model->m_RelationshipCapacity = 1;
        model->m_Relationships = new Relationship*[model->m_RelationshipCapacity];
    }
    else if(model->m_RelationshipCount >= model->m_RelationshipCapacity)
    {
        Relationship** newRelationshipArray = new Relationship*[model->m_RelationshipCapacity *2];

        std::copy(model->m_Relationships, model->m_Relationships + model->m_RelationshipCapacity, newRelationshipArray);
        delete[] model->m_Relationships;
        model->m_Relationships = newRelationshipArray;

        model->m_RelationshipCapacity = model->m_RelationshipCapacity *2;
    }

    Relationship* newRelationship = new Relationship;

    newRelationship->m_Guid = relationship.m_Guid;
    newRelationship->m_RelationshipType = relationship.m_RelationshipType;
    newRelationship->m_HeadGuid = relationship.m_HeadGuid;
    newRelationship->m_TailGuid = relationship.m_TailGuid;

    model->m_Relationships[model->m_RelationshipCount] = newRelationship;
    model->m_RelationshipCount++;
};
