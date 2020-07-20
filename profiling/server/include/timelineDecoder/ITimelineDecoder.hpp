//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdint>
#include <string>

namespace arm
{

namespace pipe
{

class ITimelineDecoder
{

public:

    enum class TimelineStatus
    {
        TimelineStatus_Success,
        TimelineStatus_Fail
    };

    enum class RelationshipType
    {
        RetentionLink, /// Head retains(parents) Tail
        ExecutionLink, /// Head execution start depends on Tail execution completion
        DataLink,      /// Head uses data of Tail
        LabelLink      /// Head uses label Tail (Tail MUST be a guid of a label).
    };

    static char const* GetRelationshipAsCString(RelationshipType rType)
    {
        switch (rType)
        {
            case RelationshipType::RetentionLink: return "RetentionLink";
            case RelationshipType::ExecutionLink: return "ExecutionLink";
            case RelationshipType::DataLink: return "DataLink";
            case RelationshipType::LabelLink: return "LabelLink";
            default: return "Unknown";
        }
    }

    struct Entity
    {
        uint64_t m_Guid;
    };

    struct EventClass
    {
        uint64_t m_Guid;
        uint64_t m_NameGuid;
    };

    struct Event
    {
        uint64_t m_Guid;
        uint64_t m_TimeStamp;
        uint64_t m_ThreadId;
    };

    struct Label
    {
        uint64_t m_Guid;
        std::string m_Name;
    };

    struct Relationship
    {
        RelationshipType m_RelationshipType;
        uint64_t m_Guid;
        uint64_t m_HeadGuid;
        uint64_t m_TailGuid;
        uint64_t m_AttributeGuid;
    };

    virtual ~ITimelineDecoder() = default;

    virtual TimelineStatus CreateEntity(const Entity&) = 0;
    virtual TimelineStatus CreateEventClass(const EventClass&) = 0;
    virtual TimelineStatus CreateEvent(const Event&) = 0;
    virtual TimelineStatus CreateLabel(const Label&) = 0;
    virtual TimelineStatus CreateRelationship(const Relationship&) = 0;
};

} // namespace pipe
} // namespace arm
