//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PerJobCounterSelectionCommandHandler.hpp"

#include <boost/format.hpp>

namespace armnn
{

namespace profiling
{

void PerJobCounterSelectionCommandHandler::operator()(const Packet& packet)
{
    ProfilingState currentState = m_StateMachine.GetCurrentState();
    switch (currentState)
    {
    case ProfilingState::Uninitialised:
    case ProfilingState::NotConnected:
    case ProfilingState::WaitingForAck:
        throw RuntimeException(boost::str(boost::format("Per-Job Counter Selection Command Handler invoked while in "
                                                        "an wrong state: %1%")
                                          % GetProfilingStateName(currentState)));
    case ProfilingState::Active:
        // Process the packet
        if (!(packet.GetPacketFamily() == 0u && packet.GetPacketId() == 5u))
        {
            throw armnn::InvalidArgumentException(boost::str(boost::format("Expected Packet family = 0, id = 5 but "
                                                                           "received family = %1%, id = %2%")
                                                  % packet.GetPacketFamily()
                                                  % packet.GetPacketId()));
        }

        // Silently drop the packet

        break;
    default:
        throw RuntimeException(boost::str(boost::format("Unknown profiling service state: %1%")
                                          % static_cast<int>(currentState)));
    }
}

} // namespace profiling

} // namespace armnn
