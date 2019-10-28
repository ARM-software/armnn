//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TimelineUtilityMethods.hpp"
#include "ProfilingService.hpp"
#include "LabelsAndEventClasses.hpp"

namespace armnn
{

namespace profiling
{

void TimelineUtilityMethods::SendWellKnownLabelsAndEventClasses()
{
    // Send the "name" label, this call throws in case of error
    m_SendTimelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::NAME_GUID,
                                                       LabelsAndEventClasses::NAME_LABEL);

    // Send the "type" label, this call throws in case of error
    m_SendTimelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::TYPE_GUID,
                                                       LabelsAndEventClasses::TYPE_LABEL);

    // Send the "index" label, this call throws in case of error
    m_SendTimelinePacket.SendTimelineLabelBinaryPacket(LabelsAndEventClasses::INDEX_GUID,
                                                       LabelsAndEventClasses::INDEX_LABEL);

    // Send the "start of life" event class, this call throws in case of error
    m_SendTimelinePacket.SendTimelineEventClassBinaryPacket(LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS);

    // Send the "end of life" event class, this call throws in case of error
    m_SendTimelinePacket.SendTimelineEventClassBinaryPacket(LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS);
}

ProfilingStaticGuid TimelineUtilityMethods::DeclareLabel(const std::string& labelName)
{
    // Check that the label name is valid
    if (labelName.empty())
    {
        // The label name is invalid
        throw InvalidArgumentException("Invalid label name, the label name cannot be empty");
    }

    // Generate a static GUID for the given label name
    ProfilingStaticGuid labelGuid = ProfilingService::Instance().GenerateStaticId(labelName);

    // Send the new label to the external profiling service, this call throws in case of error
    m_SendTimelinePacket.SendTimelineLabelBinaryPacket(labelGuid, labelName);

    return labelGuid;
}

void TimelineUtilityMethods::CreateTypedLabel(ProfilingGuid entityGuid,
                                              const std::string& entityName,
                                              ProfilingStaticGuid labelTypeGuid)
{
    // Check that the entity name is valid
    if (entityName.empty())
    {
        // The entity name is invalid
        throw InvalidArgumentException("Invalid entity name, the entity name cannot be empty");
    }

    // Declare a label with the entity's name, this call throws in case of error
    ProfilingGuid labelGuid = DeclareLabel(entityName);

    // Generate a GUID for the label relationship
    ProfilingGuid relationshipGuid = ProfilingService::Instance().NextGuid();

    // Send the new label link to the external profiling service, this call throws in case of error
    m_SendTimelinePacket.SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                                              relationshipGuid,
                                                              entityGuid,
                                                              labelGuid);

    // Generate a GUID for the label relationship
    ProfilingGuid relationshipLabelGuid = ProfilingService::Instance().NextGuid();

    // Send the new label link to the external profiling service, this call throws in case of error
    m_SendTimelinePacket.SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink,
                                                              relationshipLabelGuid,
                                                              relationshipGuid,
                                                              labelTypeGuid);
}

} // namespace profiling

} // namespace armnn
