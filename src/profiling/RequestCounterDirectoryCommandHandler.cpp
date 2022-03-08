//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RequestCounterDirectoryCommandHandler.hpp"

#include <fmt/format.h>

namespace arm
{

namespace pipe
{

void RequestCounterDirectoryCommandHandler::operator()(const arm::pipe::Packet& packet)
{
    ProfilingState currentState = m_StateMachine.GetCurrentState();
    switch (currentState)
    {
    case ProfilingState::Uninitialised:
    case ProfilingState::NotConnected:
    case ProfilingState::WaitingForAck:
        throw arm::pipe::ProfilingException(fmt::format("Request Counter Directory Comand Handler invoked while in an "
                                            "wrong state: {}",
                                            GetProfilingStateName(currentState)));
    case ProfilingState::Active:
        // Process the packet
        if (!(packet.GetPacketFamily() == 0u && packet.GetPacketId() == 3u))
        {
            throw arm::pipe::InvalidArgumentException(fmt::format("Expected Packet family = 0, id = 3 but "
                                                                  "received family = {}, id = {}",
                                                                  packet.GetPacketFamily(),
                                                                  packet.GetPacketId()));
        }

        // Send all the packet required for the handshake with the external profiling service
        m_SendCounterPacket.SendCounterDirectoryPacket(m_CounterDirectory);
        m_SendTimelinePacket.SendTimelineMessageDirectoryPackage();

        break;
    default:
        throw arm::pipe::ProfilingException(fmt::format("Unknown profiling service state: {}",
                                            static_cast<int>(currentState)));
    }
}

} // namespace pipe

} // namespace arm
