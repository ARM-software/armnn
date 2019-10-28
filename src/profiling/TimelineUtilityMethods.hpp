//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ISendTimelinePacket.hpp"
#include "ProfilingGuid.hpp"

namespace armnn
{

namespace profiling
{

class TimelineUtilityMethods
{
public:
    TimelineUtilityMethods(ISendTimelinePacket& sendTimelinePacket)
        : m_SendTimelinePacket(sendTimelinePacket)
    {}
    ~TimelineUtilityMethods() = default;

    void SendWellKnownLabelsAndEventClasses();

    ProfilingStaticGuid DeclareLabel(const std::string& labelName);
    void CreateTypedLabel(ProfilingGuid entityGuid, const std::string& entityName, ProfilingStaticGuid labelTypeGuid);

private:
    ISendTimelinePacket& m_SendTimelinePacket;
};

} // namespace profiling

} // namespace armnn
