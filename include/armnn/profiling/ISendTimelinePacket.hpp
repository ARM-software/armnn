//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace armnn
{

namespace profiling
{

enum class ProfilingRelationshipType
{
    RetentionLink,    /// Head retains(parents) Tail
    ExecutionLink,    /// Head execution start depends on Tail execution completion
    DataLink,         /// Head uses data of Tail
    LabelLink         /// Head uses label Tail (Tail MUST be a guid of a label).
};

class ISendTimelinePacket
{
public:
    virtual ~ISendTimelinePacket()
    {}

    /// Commits the current buffer and reset the member variables
    virtual void Commit() = 0;

    /// Create and write a TimelineEntityBinaryPacket from the parameters to the buffer.
    virtual void SendTimelineEntityBinaryPacket(uint64_t profilingGuid) = 0;

    /// Create and write a TimelineEventBinaryPacket from the parameters to the buffer.
    virtual void
        SendTimelineEventBinaryPacket(uint64_t timestamp, int threadId, uint64_t profilingGuid) = 0;

    /// Create and write a TimelineEventClassBinaryPacket from the parameters to the buffer.
    virtual void SendTimelineEventClassBinaryPacket(uint64_t profilingGuid, uint64_t nameGuid) = 0;

    /// Create and write a TimelineLabelBinaryPacket from the parameters to the buffer.
    virtual void SendTimelineLabelBinaryPacket(uint64_t profilingGuid, const std::string& label) = 0;

    /// Create and write a TimelineMessageDirectoryPackage in the buffer
    virtual void SendTimelineMessageDirectoryPackage() = 0;

    /// Create and write a TimelineRelationshipBinaryPacket from the parameters to the buffer.
    virtual void SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType relationshipType,
                                                      uint64_t relationshipGuid,
                                                      uint64_t headGuid,
                                                      uint64_t tailGuid,
                                                      uint64_t attributeGuid) = 0;
};

}    // namespace profiling

}    // namespace armnn
