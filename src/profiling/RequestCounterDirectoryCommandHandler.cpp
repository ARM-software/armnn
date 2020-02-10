//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RequestCounterDirectoryCommandHandler.hpp"

#include <boost/format.hpp>

namespace armnn
{

namespace profiling
{

void RequestCounterDirectoryCommandHandler::operator()(const Packet& packet)
{
    ProfilingState currentState = m_StateMachine.GetCurrentState();
    switch (currentState)
    {
    case ProfilingState::Uninitialised:
    case ProfilingState::NotConnected:
    case ProfilingState::WaitingForAck:
        throw RuntimeException(boost::str(boost::format("Request Counter Directory Comand Handler invoked while in an "
                                                        "wrong state: %1%")
                                          % GetProfilingStateName(currentState)));
    case ProfilingState::Active:
        // Process the packet
        if (!(packet.GetPacketFamily() == 0u && packet.GetPacketId() == 3u))
        {
            throw armnn::InvalidArgumentException(boost::str(boost::format("Expected Packet family = 0, id = 3 but "
                                                                           "received family = %1%, id = %2%")
                                                  % packet.GetPacketFamily()
                                                  % packet.GetPacketId()));
        }

        // Send all the packet required for the handshake with the external profiling service
        m_SendCounterPacket.SendCounterDirectoryPacket(m_CounterDirectory);
        m_SendTimelinePacket.SendTimelineMessageDirectoryPackage();

        break;
    default:
        throw RuntimeException(boost::str(boost::format("Unknown profiling service state: %1%")
                                          % static_cast<int>(currentState)));
    }
}

} // namespace profiling

} // namespace armnn
