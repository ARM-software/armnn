//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DeactivateTimelineReportingCommandHandler.hpp"

#include <armnn/Exceptions.hpp>
#include <boost/format.hpp>


namespace armnn
{

namespace profiling
{

void DeactivateTimelineReportingCommandHandler::operator()(const Packet& packet)
{
    ProfilingState currentState = m_StateMachine.GetCurrentState();

    switch ( currentState )
    {
        case ProfilingState::Uninitialised:
        case ProfilingState::NotConnected:
        case ProfilingState::WaitingForAck:
            throw RuntimeException(boost::str(
                boost::format("Deactivate Timeline Reporting Command Handler invoked while in a wrong state: %1%")
                    % GetProfilingStateName(currentState)));
        case ProfilingState::Active:
            if (!(packet.GetPacketFamily() == 0u && packet.GetPacketId() == 7u))
            {
                throw armnn::Exception(std::string("Expected Packet family = 0, id = 7 but received family =")
                                       + std::to_string(packet.GetPacketFamily())
                                       +" id = " + std::to_string(packet.GetPacketId()));
            }

            m_TimelineReporting.store(false);

            // Notify Backends
            m_BackendNotifier.NotifyBackendsForTimelineReporting();

            break;
        default:
            throw RuntimeException(boost::str(boost::format("Unknown profiling service state: %1%")
                                                  % static_cast<int>(currentState)));
    }
}

} // namespace profiling

} // namespace armnn

