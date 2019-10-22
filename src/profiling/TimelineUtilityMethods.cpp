//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TimelineUtilityMethods.hpp"
#include "ProfilingService.hpp"

namespace armnn
{

namespace profiling
{

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
