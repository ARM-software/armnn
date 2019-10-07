//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConnectionAcknowledgedCommandHandler.hpp"

#include <armnn/Exceptions.hpp>

namespace armnn
{

namespace profiling
{

void ConnectionAcknowledgedCommandHandler::operator()(const Packet& packet)
{
    if (!(packet.GetPacketFamily() == 0u && packet.GetPacketId() == 1u))
    {
        throw armnn::InvalidArgumentException(std::string("Expected Packet family = 0, id = 1 but received family = ")
                                              + std::to_string(packet.GetPacketFamily())
                                              + " id = " + std::to_string(packet.GetPacketId()));
    }

    // Once a Connection Acknowledged packet has been received, move to the Active state immediately
    m_StateMachine.TransitionToState(ProfilingState::Active);
}

} // namespace profiling

} // namespace armnn

