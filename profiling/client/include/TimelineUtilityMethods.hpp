//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <client/include/IProfilingService.hpp>
#include <client/include/ISendTimelinePacket.hpp>

namespace arm
{

namespace pipe
{

class TimelineUtilityMethods
{
public:

    // static factory method which will return a pointer to a timelie utility methods
    // object if profiling is enabled. Otherwise will return a null unique_ptr
    static std::unique_ptr<TimelineUtilityMethods> GetTimelineUtils(IProfilingService& profilingService);

    TimelineUtilityMethods(
        std::unique_ptr<ISendTimelinePacket>& sendTimelinePacket)
        : m_SendTimelinePacket(std::move(sendTimelinePacket)) {}

    TimelineUtilityMethods(TimelineUtilityMethods&& other)
        : m_SendTimelinePacket(std::move(other.m_SendTimelinePacket)) {}

    TimelineUtilityMethods(const TimelineUtilityMethods& other) = delete;

    TimelineUtilityMethods& operator=(const TimelineUtilityMethods& other) = delete;

    TimelineUtilityMethods& operator=(TimelineUtilityMethods&& other) = default;

    ~TimelineUtilityMethods() = default;

    static void SendWellKnownLabelsAndEventClasses(ISendTimelinePacket& timelinePacket);

    ProfilingDynamicGuid CreateNamedTypedEntity(const std::string& name, const std::string& type);

    void CreateNamedTypedEntity(ProfilingGuid entityGuid, const std::string& name, const std::string& type);

    void CreateNamedTypedEntity(ProfilingGuid entityGuid, const std::string& name, ProfilingStaticGuid typeGuid);

    void MarkEntityWithLabel(ProfilingGuid entityGuid, const std::string& labelName, ProfilingStaticGuid labelLinkGuid);

    ProfilingStaticGuid DeclareLabel(const std::string& labelName);

    void NameEntity(ProfilingGuid entityGuid, const std::string& name);

    void TypeEntity(ProfilingGuid entityGuid, const std::string& type);

    ProfilingDynamicGuid CreateNamedTypedChildEntity(ProfilingGuid parentEntityGuid,
                                                     const std::string& entityName,
                                                     const std::string& entityType);

    void CreateNamedTypedChildEntity(ProfilingGuid entityGuid,
                                     ProfilingGuid parentEntityGuid,
                                     const std::string& entityName,
                                     const std::string& entityType);

    void CreateNamedTypedChildEntity(ProfilingGuid entityGuid,
                                     ProfilingGuid parentEntityGuid,
                                     const std::string& entityName,
                                     ProfilingStaticGuid typeGuid);

    ProfilingDynamicGuid CreateRelationship(ProfilingRelationshipType relationshipType,
                                            ProfilingGuid headGuid,
                                            ProfilingGuid tailGuid,
                                            ProfilingGuid relationshipCategory);

    ProfilingDynamicGuid CreateConnectionRelationship(ProfilingRelationshipType relationshipType,
                                                      ProfilingGuid headGuid,
                                                      ProfilingGuid tailGuid);

    void CreateTypedEntity(ProfilingGuid entityGuid, ProfilingStaticGuid typeGuid);

    void MarkEntityWithType(ProfilingGuid entityGuid, ProfilingStaticGuid typeNameGuid);

    ProfilingDynamicGuid RecordEvent(ProfilingGuid entityGuid, ProfilingStaticGuid eventClassGuid);

    ProfilingDynamicGuid RecordWorkloadInferenceAndStartOfLifeEvent(ProfilingGuid workloadGuid,
                                                                    ProfilingGuid inferenceGuid);

    void RecordEndOfLifeEvent(ProfilingGuid entityGuid);

    void Commit() { m_SendTimelinePacket->Commit(); }

private:
    std::unique_ptr<ISendTimelinePacket> m_SendTimelinePacket;
};

} // namespace pipe

} // namespace arm
