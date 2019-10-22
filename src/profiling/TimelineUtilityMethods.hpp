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

    ProfilingStaticGuid DeclareLabel(const std::string& labelName);

private:
    ISendTimelinePacket& m_SendTimelinePacket;
};

} // namespace profiling

} // namespace armnn
