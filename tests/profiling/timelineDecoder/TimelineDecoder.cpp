//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ITimelineDecoder.h"

ErrorCode CreateEntity(const Entity entity, Model* model)
{
    if (model == nullptr || model->m_EntityCb == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    model->m_EntityCb(entity, model);
    return ErrorCode::ErrorCode_Success;
}

ErrorCode CreateEventClass(const EventClass eventClass, Model* model)
{
    if (model == nullptr || model->m_EventClassCb == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    model->m_EventClassCb(eventClass, model);
    return ErrorCode::ErrorCode_Success;
}

ErrorCode CreateEvent(const Event event, Model* model)
{
    if (model == nullptr || model->m_EventCb == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    model->m_EventCb(event, model);
    return ErrorCode::ErrorCode_Success;
}

ErrorCode CreateLabel(const Label label, Model* model)
{
    if (model == nullptr || model->m_LabelCb == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    model->m_LabelCb(label, model);
    return ErrorCode::ErrorCode_Success;
}

ErrorCode CreateRelationship(Relationship relationship, Model* model)
{
    if (model == nullptr || model->m_RelationshipCb == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    model->m_RelationshipCb(relationship, model);
    return ErrorCode::ErrorCode_Success;
}

ErrorCode SetEntityCallback(OnNewEntityCallback cb, Model* model)
{
    if (cb == nullptr || model == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    model->m_EntityCb = cb;
    return ErrorCode::ErrorCode_Success;
}

ErrorCode SetEventClassCallback(OnNewEventClassCallback cb, Model* model)
{
    if (cb == nullptr || model == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    model->m_EventClassCb = cb;
    return ErrorCode::ErrorCode_Success;
}

ErrorCode SetEventCallback(OnNewEventCallback cb, Model* model)
{
    if (cb == nullptr || model == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    model->m_EventCb = cb;
    return ErrorCode::ErrorCode_Success;
}

ErrorCode SetLabelCallback(OnNewLabelCallback cb, Model* model)
{
    if (cb == nullptr || model == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    model->m_LabelCb = cb;
    return ErrorCode::ErrorCode_Success;
}

ErrorCode SetRelationshipCallback(OnNewRelationshipCallback cb, Model* model)
{
    if (cb == nullptr || model == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }
    model->m_RelationshipCb = cb;
    return ErrorCode::ErrorCode_Success;
}

ErrorCode CreateModel(Model** model)
{
    Model* modelPtr = new Model;

    modelPtr->m_EntityCount = 0;
    modelPtr->m_EventClassCount = 0;
    modelPtr->m_EventCount = 0;
    modelPtr->m_LabelCount = 0;
    modelPtr->m_RelationshipCount = 0;

    *model = modelPtr;
    return ErrorCode::ErrorCode_Success;
}

ErrorCode DestroyModel(Model** model)
{
    if (*model == nullptr)
    {
        return ErrorCode::ErrorCode_Fail;
    }

    Model* modelPtr = *model;

    for (uint32_t i = 0; i < modelPtr->m_EntityCount; ++i)
    {
        delete modelPtr->m_Entities[i];
    }

    for (uint32_t i = 0; i < modelPtr->m_EventClassCount; ++i)
    {
        delete modelPtr->m_EventClasses[i];
    }

    for (uint32_t i = 0; i < modelPtr->m_EventCount; ++i)
    {
        delete[] modelPtr->m_Events[i]->m_ThreadId;
        delete modelPtr->m_Events[i];
    }

    for (uint32_t i = 0; i < modelPtr->m_LabelCount; ++i)
    {
        delete[] modelPtr->m_Labels[i]->m_Name;
        delete modelPtr->m_Labels[i];
    }

    for (uint32_t i = 0; i < modelPtr->m_RelationshipCount; ++i)
    {
        delete modelPtr->m_Relationships[i];
    }

    delete[] modelPtr->m_Entities;
    delete[] modelPtr->m_EventClasses;
    delete[] modelPtr->m_Events;
    delete[] modelPtr->m_Labels;
    delete[] modelPtr->m_Relationships;

    delete modelPtr;
    return ErrorCode::ErrorCode_Success;
}