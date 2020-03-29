//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TimelineModel.hpp"

namespace armnn
{

namespace profiling
{

void TimelineModel::AddLabel(const ITimelineDecoder::Label& label)
{
    m_LabelMap.emplace(label.m_Guid, label);
}

void TimelineModel::AddEntity(uint64_t guid)
{
    m_Entities.emplace(guid, guid);
}

Entity* TimelineModel::findEntity(uint64_t id)
{
    auto iter = m_Entities.find(id);
    if (iter != m_Entities.end())
    {
        return &(iter->second);
    }
    else
    {
        return nullptr;
    }
}

void TimelineModel::AddRelationship(const ITimelineDecoder::Relationship& relationship)
{
    m_Relationships.emplace(relationship.m_Guid, relationship);
}

ModelRelationship* TimelineModel::findRelationship(uint64_t id)
{
    auto iter = m_Relationships.find(id);
    if (iter != m_Relationships.end())
    {
        return &(iter->second);
    }
    else
    {
        return nullptr;
    }
}

} // namespace profiling

} // namespace armnn