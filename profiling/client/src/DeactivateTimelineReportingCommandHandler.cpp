//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DeactivateTimelineReportingCommandHandler.hpp"

#include <common/include/ProfilingException.hpp>

#include <fmt/format.h>


namespace arm
{

namespace pipe
{

void DeactivateTimelineReportingCommandHandler::operator()(const arm::pipe::Packet& packet)
{
    ProfilingState currentState = m_StateMachine.GetCurrentState();

    switch ( currentState )
    {
        case ProfilingState::Uninitialised:
        case ProfilingState::NotConnected:
        case ProfilingState::WaitingForAck:
            throw arm::pipe::ProfilingException(fmt::format(
                "Deactivate Timeline Reporting Command Handler invoked while in a wrong state: {}",
                GetProfilingStateName(currentState)));
        case ProfilingState::Active:
            if (!(packet.GetPacketFamily() == 0u && packet.GetPacketId() == 7u))
            {
                throw arm::pipe::ProfilingException(std::string(
                    "Expected Packet family = 0, id = 7 but received family =")
                    + std::to_string(packet.GetPacketFamily())
                    + " id = " + std::to_string(packet.GetPacketId()));
            }

            m_TimelineReporting.store(false);

            // Notify Backends
            m_BackendNotifier.NotifyBackendsForTimelineReporting();

            break;
        default:
            throw arm::pipe::ProfilingException(fmt::format("Unknown profiling service state: {}",
                                                static_cast<int>(currentState)));
    }
}

} // namespace pipe

} // namespace arm
