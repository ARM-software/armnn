//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#ifndef ARMNN_ITIMELINEDECODER_H
#define ARMNN_ITIMELINEDECODER_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "TimelineModel.h"

typedef enum ErrorCode
{
    ErrorCode_Success,
    ErrorCode_Fail
} ErrorCode;

ErrorCode CreateModel(Model** model);
ErrorCode DestroyModel(Model** model);

ErrorCode SetEntityCallback(OnNewEntityCallback cb, Model* model);
ErrorCode SetEventClassCallback(OnNewEventClassCallback cb, Model* model);
ErrorCode SetEventCallback(OnNewEventCallback cb, Model* model);
ErrorCode SetLabelCallback(OnNewLabelCallback cb, Model* model);
ErrorCode SetRelationshipCallback(OnNewRelationshipCallback cb, Model* model);

ErrorCode CreateEntity(const Entity entity, Model* model);
ErrorCode CreateEventClass(const EventClass eventClass, Model* model);
ErrorCode CreateEvent(const Event event, Model* model);
ErrorCode CreateLabel(const Label label, Model* model);
ErrorCode CreateRelationship(const Relationship relationship, Model* model);

#ifdef __cplusplus
}
#endif

#endif //ARMNN_ITIMELINEDECODER_H