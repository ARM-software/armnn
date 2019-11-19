//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ISendTimelinePacket.hpp"
#include <armnn/Types.hpp>

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

    ProfilingDynamicGuid CreateNamedTypedEntity(const std::string& name, const std::string& type);

    void CreateNamedTypedEntity(ProfilingDynamicGuid entityGuid, const std::string& name, const std::string& type);

    void CreateTypedLabel(ProfilingGuid entityGuid, const std::string& entityName, ProfilingStaticGuid labelTypeGuid);

    ProfilingStaticGuid DeclareLabel(const std::string& labelName);

    void NameEntity(ProfilingGuid entityGuid, const std::string& name);

    void TypeEntity(ProfilingGuid entityGuid, const std::string& type);

    ProfilingDynamicGuid CreateNamedTypedChildEntity(ProfilingGuid parentEntityGuid,
                                                     const std::string& entityName,
                                                     const std::string& entityType);

    void CreateNamedTypedChildEntity(ProfilingDynamicGuid entityGuid,
                                     ProfilingGuid parentEntityGuid,
                                     const std::string& entityName,
                                     const std::string& entityType);

    ProfilingDynamicGuid RecordEvent(ProfilingGuid entityGuid, ProfilingStaticGuid eventClassGuid);

private:
    ISendTimelinePacket& m_SendTimelinePacket;
};

} // namespace profiling

} // namespace armnn
