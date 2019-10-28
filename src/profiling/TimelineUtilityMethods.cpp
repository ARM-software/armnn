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

} // namespace profiling

} // namespace armnn
