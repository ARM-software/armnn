//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#ifndef ARMNN_ITIMELINEMODEL_H
#define ARMNN_ITIMELINEMODEL_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>

struct Model;

typedef enum RelationshipType
{
    RetentionLink, /// Head retains(parents) Tail
    ExecutionLink, /// Head execution start depends on Tail execution completion
    DataLink,      /// Head uses data of Tail
    LabelLink      /// Head uses label Tail (Tail MUST be a guid of a label).
} RelationshipType;

typedef struct Entity
{
    uint64_t m_Guid;
} Entity;

typedef struct EventClass
{
    uint64_t m_Guid;
} EventClass;

typedef struct Event
{
    uint64_t m_Guid;
    uint64_t m_TimeStamp;
    unsigned char* m_ThreadId;
} ProfilingEvent;

typedef struct Label
{
    uint64_t m_Guid;
    char* m_Name;
} Label;

typedef struct Relationship
{
    RelationshipType m_RelationshipType;
    uint64_t m_Guid;
    uint64_t m_HeadGuid;
    uint64_t m_TailGuid;
} Relationship;

typedef void (*OnNewEntityCallback)(const Entity, struct Model* model);
typedef void (*OnNewEventClassCallback)(const EventClass, struct Model* model);
typedef void (*OnNewEventCallback)(const Event, struct Model* model);
typedef void (*OnNewLabelCallback)(const Label, struct Model* model);
typedef void (*OnNewRelationshipCallback)(const Relationship, struct Model* model) ;

typedef struct Model
{
    OnNewEntityCallback m_EntityCb;
    OnNewEventClassCallback m_EventClassCb;
    OnNewEventCallback m_EventCb;
    OnNewLabelCallback m_LabelCb;
    OnNewRelationshipCallback m_RelationshipCb;

    Entity** m_Entities;
    EventClass** m_EventClasses;
    Event** m_Events;
    Label** m_Labels;
    Relationship** m_Relationships;

    uint32_t m_EntityCount;
    uint32_t m_EntityCapacity;

    uint32_t m_EventClassCount;
    uint32_t m_EventClassCapacity;

    uint32_t m_EventCount;
    uint32_t m_EventCapacity;

    uint32_t m_LabelCount;
    uint32_t m_LabelCapacity;

    uint32_t m_RelationshipCount;
    uint32_t m_RelationshipCapacity;
} Model;

#ifdef __cplusplus
}
#endif

#endif //ARMNN_ITIMELINEMODEL_H