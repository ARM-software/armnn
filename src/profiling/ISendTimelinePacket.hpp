//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ProfilingUtils.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace armnn
{

namespace profiling
{

class ISendTimelinePacket
{
public:
    virtual ~ISendTimelinePacket() {}

    /// Commits the current buffer and reset the member variables
    virtual void Commit() = 0;

    /// Create and write a TimelineEntityBinaryPacket from the parameters to the buffer.
    virtual void SendTimelineEntityBinaryPacket(uint64_t profilingGuid) = 0;

    /// Create and write a TimelineEventBinaryPacket from the parameters to the buffer.
    virtual void SendTimelineEventBinaryPacket(uint64_t timestamp, uint32_t threadId, uint64_t profilingGuid) = 0;

    /// Create and write a TimelineEventClassBinaryPacket from the parameters to the buffer.
    virtual void SendTimelineEventClassBinaryPacket(uint64_t profilingGuid) = 0;

    /// Create and write a TimelineLabelBinaryPacket from the parameters to the buffer.
    virtual void SendTimelineLabelBinaryPacket(uint64_t profilingGuid, const std::string& label) = 0;

    /// Create and write a TimelineMessageDirectoryPackage in the buffer
    virtual void SendTimelineMessageDirectoryPackage() = 0;

    /// Create and write a TimelineRelationshipBinaryPacket from the parameters to the buffer.
    virtual void SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType relationshipType,
                                                      uint64_t relationshipGuid,
                                                      uint64_t headGuid,
                                                      uint64_t tailGuid) = 0;
};

} // namespace profiling

} // namespace armnn

