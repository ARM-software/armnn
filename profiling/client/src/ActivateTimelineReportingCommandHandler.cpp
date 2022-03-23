//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ActivateTimelineReportingCommandHandler.hpp"

#include <client/include/TimelineUtilityMethods.hpp>

#include <common/include/ProfilingException.hpp>

#include <fmt/format.h>

namespace arm
{

namespace pipe
{

void ActivateTimelineReportingCommandHandler::operator()(const arm::pipe::Packet& packet)
{
    ProfilingState currentState = m_StateMachine.GetCurrentState();

    if (!m_ReportStructure.has_value())
    {
        throw arm::pipe::ProfilingException(std::string(
                                                "Profiling Service constructor must be initialised with an "
                                                "IReportStructure argument in order to run timeline reporting"));
    }

    switch ( currentState )
    {
        case ProfilingState::Uninitialised:
        case ProfilingState::NotConnected:
        case ProfilingState::WaitingForAck:
            throw arm::pipe::ProfilingException(fmt::format(
                    "Activate Timeline Reporting Command Handler invoked while in a wrong state: {}",
                    GetProfilingStateName(currentState)));
        case ProfilingState::Active:
            if ( !( packet.GetPacketFamily() == 0u && packet.GetPacketId() == 6u ))
            {
                throw arm::pipe::ProfilingException(std::string(
                                           "Expected Packet family = 0, id = 6 but received family =")
                                           + std::to_string(packet.GetPacketFamily())
                                           + " id = " + std::to_string(packet.GetPacketId()));
            }

            if (!m_TimelineReporting)
            {
                m_SendTimelinePacket.SendTimelineMessageDirectoryPackage();

                TimelineUtilityMethods::SendWellKnownLabelsAndEventClasses(m_SendTimelinePacket);

                m_TimelineReporting = true;

                if (m_ReportStructure.has_value())
                {
                    m_ReportStructure.value().ReportStructure(m_ProfilingService);
                }

                m_BackendNotifier.NotifyBackendsForTimelineReporting();
            }

            break;
        default:
            throw arm::pipe::ProfilingException(fmt::format("Unknown profiling service state: {}",
                                                static_cast<int>(currentState)));
    }
}

} // namespace pipe

} // namespace arm
